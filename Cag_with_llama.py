import streamlit as st
from functools import lru_cache
import pymupdf as fitz
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
import hashlib
import os
import pickle
from PIL import Image
import io
import requests
import tempfile
import base64

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

qdrant_client = QdrantClient(":memory:")
client = OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1"
)

def get_cache_file_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def cache_get(key: str):
    try:
        with open(get_cache_file_path(key), 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None

def cache_set(key: str, value):
    with open(get_cache_file_path(key), 'wb') as f:
        pickle.dump(value, f)

def cache_clear():
    for file in os.listdir(CACHE_DIR):
        os.remove(os.path.join(CACHE_DIR, file))

qdrant_client.create_collection(
    collection_name="Sakthi's_vector_store",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
)

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_tokens:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@lru_cache(maxsize=100)
def get_cached_embedding(text_chunk: str):
    try:
        response = client.embeddings.create(
            input=text_chunk,
            model="nvidia/nv-embedqa-mistral-7b-v2",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

def get_cache_key(query: str, file_content: str) -> str:
    combined = query + file_content[:1000]
    return hashlib.md5(combined.encode()).hexdigest()

def process_file(file):
    text = ""
    if file is None:
        return "No file uploaded"
    
    doc = fitz.open("pdf", file.read())
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# New image analysis function using llama-3.2-90b-vision-instruct
def analyze_image_with_llama_vision(query: str, image_bytes: bytes):
    try:
        # Convert image to base64
        image_b64 = base64.b64encode(image_bytes).decode()
        

            
        headers = {
            "Authorization": "Bearer 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",
            "Accept": "application/json"
        }
        
        payload = {
            "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "messages": [
                {
                    "role": "user",
                    "content": f'{query} <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
        
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
    except Exception as e:
        st.error(f"Image analysis error: {str(e)}")
        return None

def main():
    st.title("Document Analysis System")
    
    tab1, tab2 = st.tabs(["Text Query", "Image Analysis"])
    
    with tab1:
        st.header("Text-Based Query")
        file = st.file_uploader("Upload PDF File", type=["pdf"], key="file_uploader")
        file_content = ""
        
        if file is not None:
            with st.spinner("Processing document..."):
                file_content = process_file(file)
                if file_content:
                    text_chunks = chunk_text(file_content)
                    points = []
                    for idx, chunk in enumerate(text_chunks[:5]):
                        embedding = get_cached_embedding(chunk)
                        if embedding is not None:
                            points.append(
                                PointStruct(
                                    id=idx,
                                    vector=embedding,
                                    payload={"text": chunk}
                                )
                            )
                    if points:
                        qdrant_client.upsert(
                            collection_name="Sakthi's_vector_store",
                            points=points
                        )
                    st.success("Document processed successfully")

        text_query = st.text_input("Ask about the document content:")
        
        if text_query and file_content:
            cache_key = f"text_{get_cache_key(text_query, file_content)}"
            cached_answer = cache_get(cache_key)
            
            if cached_answer is not None:
                st.markdown("### Answer:")
                st.markdown(cached_answer)
            else:
                with st.spinner("Searching document..."):
                    query_embedding = get_cached_embedding(text_query)
                    if query_embedding is not None:
                        results = qdrant_client.search(
                            collection_name="Sakthi's_vector_store",
                            query_vector=query_embedding,
                            limit=3
                        )
                        
                        if results:
                            context = "\n\n".join([hit.payload["text"] for hit in results])
                            
                            completion = client.chat.completions.create(
                                model="nvidia/llama-3.3-nemotron-super-49b-v1",
                                messages=[
                                    {"role": "system", "content": "Answer strictly based on the context."},
                                    {"role": "user", "content": f"Question: {text_query}\nContext: {context}"}
                                ],
                                temperature=0.1
                            )
                            
                            full_response = completion.choices[0].message.content
                            st.markdown("### Answer:")
                            st.markdown(full_response)
                            cache_set(cache_key, full_response)
    
    with tab2:
        st.header("Image Analysis")
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        image_query = st.text_input("What would you like to know about this image?")
        
        if uploaded_image is not None and image_query:
            cache_key = f"image_{hashlib.md5(image_query.encode()).hexdigest()}"
            cached_answer = cache_get(cache_key)
            
            if cached_answer is not None:
                st.image(Image.open(io.BytesIO(uploaded_image.getvalue())) if isinstance(uploaded_image, io.BytesIO) else uploaded_image)
                st.markdown("### Analysis:")
                st.markdown(cached_answer)
            else:
                with st.spinner("Analyzing image..."):
                    image_bytes = uploaded_image.read()
                    analysis = analyze_image_with_llama_vision(image_query, image_bytes)
                    
                    if analysis:
                        st.image(Image.open(io.BytesIO(image_bytes)))
                        st.markdown("### Analysis:")
                        st.markdown(analysis)
                        cache_set(cache_key, analysis)
                    else:
                        st.error("Failed to analyze the image")

    if st.button("Clear Cache"):
        cache_clear()
        get_cached_embedding.cache_clear()
        st.success("Cache cleared!")

if __name__ == "__main__":
    main()