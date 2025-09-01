# THE librarian

acest proiect implementeaza un chatbot care recomanda carti in 2 pasi. Folosește RAG: transforma întrebarea utilizatorului in embeddings OpenAI și caută cele mai relevante descrieri în baza locala ChromaDB. Apoi selecteaza o singura carte și apelează tool-ul get_summary_by_title, care furnizeaza un rezumat detaliat al titlului ales, pe care il inserează in raspunsul final

Proiectul include:
- interfata CLI simplă (`chatbot_cli.py`)
- suport optional pentru Streamlit (UI web)
- suport optional pentru text-to-speech (pyttsx3, Windows)

---

## functionalitati
- baza de date cu peste 10 cărți (`data/book_summaries.md` și `data/book_summaries.json`)
- incarcare în chromadb cu embeddings OpenAI (`app/rag_init.py`)
- retriever semantic pentru recomandări
- function calling OpenAI pentru tool-ul `get_summary_by_title`
- filtru simplu de limbaj nepotrivit (`app/utils.py`)
- suport pentru TTS pe Windows
- interfata web cu Streamlit (`app/st_app.py`)

---
