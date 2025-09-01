import json
from pathlib import Path
from typing import Dict


DATA_JSON = Path(__file__).resolve().parent.parent / "data" / "book_summaries.json"

def load_book_summaries_dict() -> Dict[str, str]:
    """
    incarca dictionarul cu rezumatele detaliate din json
    """
    data = json.loads(Path(DATA_JSON).read_text(encoding="utf-8"))
    # normalizare chei to lowercase pt cautare completa
    normalized = {}
    for k, v in data.items():
        normalized[k.lower()] = v
    return normalized

BOOK_DICT = load_book_summaries_dict()

def get_summary_by_title(title: str) -> str:
    """
    cauta titlul (case-insensitive) si intoarce rezumatul detaliat
    daca nu exista, intoarce un mesaj corespunzator
    """
    if not title:
        return "nu am primit un titlu valid."
    key = title.lower().strip()
    return BOOK_DICT.get(key, "nu am gasit un rezumat detaliat pentru acest titlu.")
