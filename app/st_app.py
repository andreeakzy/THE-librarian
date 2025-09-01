import streamlit as st
from dotenv import load_dotenv
from .utils import retrieve_books, get_models, get_openai_client, profanity_found
from .tools import get_summary_by_title
from openai import OpenAI

SYSTEM_PROMPT = (
    "esti un asistent bibliotecar. primesti intrebari despre carti si interesele de lectura ale utilizatorului. "
    "ai primit deja mai jos o lista de recomandari posibile de carti (titlu si rezumat scurt) care s-ar potrivi. "
    "alege o singura carte care se potriveste cel mai bine utilizatorului, explica pe scurt de ce. "
    "apoi apeleaza intotdeauna tool-ul 'get_summary_by_title' cu titlul exact recomandat, "
    "ca sa aduci un rezumat detaliat. dupa ce primesti rezultatul tool-ului, combina-l intr-un raspuns final clar. "
    "daca utilizatorul cere ceva ofensator sau nepotrivit, refuza politicos. "
)

def build_tools_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_summary_by_title",
                "description": "intoarce un rezumat detaliat pentru un titlu exact de carte",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            },
        }
    ]

def main():
    load_dotenv()
    st.set_page_config(page_title="smart-librarian", page_icon="📚")
    st.title("smart-librarian · rag + tool calling")    

    chat_model, _ = get_models()
    client = get_openai_client()
    tools_schema = build_tools_schema()

    user_q = st.text_input("AICI pune o intrebare sau exprima o preferinta literara:")
    if st.button("cauta si recomanda") and user_q:
        if profanity_found(user_q):
            st.warning("te rog foloseste un limbaj potrivit :))")
            return

        cands = retrieve_books(user_q, k=3)
        with st.expander("optiuni din retriever", expanded=True):
            for c in cands:
                st.markdown(f"**{c['title']}**\n\n{c['summary']}\n\n_distanta: {c['distance']:.4f}_")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_q},
            {"role": "system", "content": "optiuni din retriever (top 3):\n" + "\n".join([f"- {c['title']} -> {c['summary']}" for c in cands])},
        ]

        resp = client.chat.completions.create(model=chat_model, messages=messages, tools=tools_schema, tool_choice="auto", temperature=0.3)
        msg = resp.choices[0].message

        final_text = msg.content or ""
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function and tc.function.name == "get_summary_by_title":
                    import json as _json
                    title = _json.loads(tc.function.arguments or "{}").get("title", "")
                    tool_result = get_summary_by_title(title)
                    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": "get_summary_by_title", "content": tool_result})
                    resp2 = client.chat.completions.create(model=chat_model, messages=messages, temperature=0.3)
                    final_text = resp2.choices[0].message.content

        st.markdown("### raspuns")        
        st.write(final_text)

if __name__ == "__main__":
    main()
