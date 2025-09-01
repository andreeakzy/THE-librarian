import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
import chromadb
from .utils import OpenAIEmbedder, get_models, get_chroma_collection

MD_PATH = Path(__file__).resolve().parent.parent / "data" / "book_summaries.md"

def parse_markdown(md_text: str) -> List[Tuple[str, str]]:
    lines = md_text.splitlines()
    entries = []
    current_title = None
    buffer = []
    pat = re.compile(r"^##\s*title:\s*(.+)$", re.IGNORECASE)
    for line in lines + ["## end: end"]: 
        m = pat.match(line.strip())
        if m:
            if current_title and buffer:
                entries.append((current_title.strip(), " ".join(b.strip() for b in buffer).strip()))
                buffer = []
            current_title = m.group(1)
        else:
            if current_title:
                if line.strip():
                    buffer.append(line.strip())
    # curatare si validare
    entries = [(t, s) for (t, s) in entries if t and s]
    return entries

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def main(recreate: bool = False):
    load_dotenv()
    chroma_path = os.getenv("CHROMA_PATH", "./chroma")
    coll = get_chroma_collection(chroma_path)

    if recreate:
        client = chromadb.PersistentClient(path=chroma_path)
        client.delete_collection(coll.name)
        coll = client.get_or_create_collection(coll.name, metadata={"hnsw:space": "cosine"})

    md_text = Path(MD_PATH).read_text(encoding="utf-8")
    entries = parse_markdown(md_text)
    if not entries:
        print("nu am gasit intrari in markdown")
        return

    client = OpenAI()
    _, emb_model = get_models()
    embedder = OpenAIEmbedder(client, emb_model)

    ids = [slugify(t) for t, _ in entries]
    docs = [s for _, s in entries]
    metas = [{"title": t} for t, _ in entries]

    # upsert in chromadb
    coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embedder(docs))
    print(f"ingestat {len(ids)} intrari in colectia '{coll.name}' la '{chroma_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="sterge si recreeaza colectia inainte de ingestie")
    args = parser.parse_args()
    main(recreate=args.recreate)
