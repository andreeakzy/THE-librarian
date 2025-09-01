import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

def get_openai_client() -> OpenAI:
    """
    creeaza clientul openai folosind cheia din env var
    """
    return OpenAI()

def get_models() -> Tuple[str, str]:
    """
    returneaza (chat_model, embedding_model) din env sau valorile default
    """
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    emb_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return chat_model, emb_model

class OpenAIEmbedder:
    """
    wrapper pt embeddings de la openai compatibile cu chromadb
    """
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

def get_chroma_collection(path: str, collection_name: str = "book_summaries"):
    """
    creeaza sau deschide o colectie chromadb 
    """
    client = chromadb.PersistentClient(path=path)
    coll = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    return coll

def profanity_found(text: str) -> bool:
    """
    filtru minimal pt limbajul nepotrivit 
    """
    bad_words = [
        r"\bidiot\b", r"\bprost\b", r"\bnaiba\b", r"\bdracu\b",
    ]
    lower = text.lower()
    return any(re.search(pat, lower) for pat in bad_words)

def retrieve_books(question: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    cauta semantic in chromadb cele mai relevante k rezumate pentru intrebare
    """
    chroma_path = os.getenv("CHROMA_PATH", "./chroma")
    coll = get_chroma_collection(chroma_path)
    client = get_openai_client()
    _, emb_model = get_models()
    embedder = OpenAIEmbedder(client, emb_model)

    res = coll.query(
        query_texts=[question],
        n_results=k,
        embedding_function=embedder,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    # transforma raspunsul in lista de rezultate
    out = []
    if res and res.get("ids") and res["ids"] and res["ids"][0]:
        for i, _id in enumerate(res["ids"][0]):
            item = {
                "id": _id,
                "title": res["metadatas"][0][i].get("title", "unknown"),
                "summary": res["documents"][0][i],
                "distance": res["distances"][0][i],
            }
            out.append(item)
    return out


