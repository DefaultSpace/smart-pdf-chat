import os
import faiss
import pickle
from langchain_ollama import OllamaEmbeddings # Yeni import
from langchain_community.vectorstores import FAISS # FAISS importu zaten doğru olmalı

def embed_and_store(documents, db_path="vectordb/db.faiss"): # chunks parametresini documents olarak değiştirdik
    embeddings = OllamaEmbeddings(model="qwen2.5:latest")
    # Document nesneleri listesi için from_documents kullanılır
    vectorstore = FAISS.from_documents(documents, embedding=embeddings) 
    vectorstore.save_local(db_path)

def load_vectorstore(db_path="vectordb/db.faiss"):
    embeddings = OllamaEmbeddings(model="qwen2.5:latest") # Bu da yeni importu kullanacak
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
