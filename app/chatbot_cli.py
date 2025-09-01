import argparse
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from .utils import get_models, get_openai_client, retrieve_books, profanity_found
from .tools import get_summary_by_title
from .tts import speak


SYSTEM_PROMPT = (
    "esti un asistent bibliotecar. primesti intrebari despre carti si interesele de lectura ale utilizatorului. "
    "ai primit deja mai jos o lista de recomandari posibile de carti (titlu si rezumat scurt) care s-ar potrivi. "
    "alege o singura carte care se potriveste cel mai bine utilizatorului, explica pe scurt de ce. "
    "apoi apeleaza intotdeauna tool-ul 'get_summary_by_title' cu titlul exact recomandat, "
    "ca sa aduci un rezumat detaliat. dupa ce primesti rezultatul tool-ului, combina-l intr-un raspuns final clar. "
    "daca utilizatorul cere ceva ofensator sau nepotrivit, refuza politicos. "
)

def build_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_summary_by_title",
                "description": "intoarce un rezumat detaliat pentru un titlu exact de carte",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "titlul exact al cartii"}
                    },
                    "required": ["title"],
                },
            },
        }
    ]

def format_candidates(cands: List[Dict[str, Any]]) -> str:
    # formateaza top k rezultate ale retrieverului pentru a le oferi llmului
    lines = []
    for i, c in enumerate(cands, start=1):
        lines.append(f"{i}. {c['title']} -> {c['summary']} (distanta={c['distance']:.4f})")
    return "\n".join(lines) if lines else "(fara rezultate din retriever)"

def run_cli(speak_out: bool = False):
    load_dotenv()
    client = get_openai_client()
    chat_model, _ = get_models()
    tools_schema = build_tools_schema()

    print("smart-librarian. daca vrei sa iesi scrie 'exit'\n")
    while True:
        user_q = input("tu: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit", "q"}:
            print("bye!")
            break

        # filtru pt limbaj nepotrivit
        if profanity_found(user_q):
            print("asistent: inteleg frustrarea, dar te rog sa folosim un limbaj potrivit. cum te pot ajuta cu recomandari de carti?")
            continue

        # 1) retriever semantic
        cands = retrieve_books(user_q, k=3)
        context_block = "candidati din retriever (top 3):\n" + format_candidates(cands)

        # 2) conversatie catre llm (cu tool calling)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_q},
            {"role": "system", "content": context_block},
        ]

        resp = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = resp.choices[0].message

        # 3) daca llm cere apelul toolului se executa local si se continua conversatia
        final_text = None
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function and tc.function.name == "get_summary_by_title":
                    import json as _json
                    args = _json.loads(tc.function.arguments or "{}")
                    title = args.get("title", "")
                    tool_result = get_summary_by_title(title)
                    # trimite un mesaj catre llm cu rezultatul toolului
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "get_summary_by_title",
                        "content": tool_result,
                    }
                    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc]})
                    messages.append(tool_msg)
                    resp2 = client.chat.completions.create(
                        model=chat_model,
                        messages=messages,
                        temperature=0.3,
                    )
                    final_text = resp2.choices[0].message.content
        else:
            final_text = msg.content or "nu am putut genera un raspuns."

        print(f"asistent: {final_text}\n")
        if speak_out:
            try:
                speak(final_text, save_to_wav=False)
            except Exception as e:
                print(f"[tts] nu am reusit sa redau audio: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speak", action="store_true", help="citeste raspunsul cu voce (pyttsx3)" )
    args = parser.parse_args()
    run_cli(speak_out=args.speak)
