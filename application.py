import pandas as pd
import numpy as np
import faiss
import re
import os
import pickle
import json
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from collections import Counter
import requests

st.set_page_config(page_title="Library Assistant", layout="wide")

# === Load Models ===
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    base_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    finetuned_generator = pipeline("text2text-generation", model="./finetuned-flan-t5")
    return embedder, cross_encoder, base_generator, finetuned_generator

embedder, cross_encoder, base_generator, finetuned_generator = load_models()

# === Ollama Generator ===
def ollama_generate(prompt, model="gemma3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.ok:
            return response.json().get("response", "").strip()
        else:
            return f"[Ollama Error] {response.status_code}: {response.text}"
    except Exception as e:
        return f"[Ollama Exception] {str(e)}"

# === Load Exact QA Dataset ===
with open("diverse_qa_pairs.json", "r", encoding="utf-8") as f:
    qa_50_data = json.load(f)
exact_qa_dict = {entry["question"].strip().lower(): entry for entry in qa_50_data}

# === Load Data & Build FAISS Index ===
@st.cache_data
def load_data_and_index():
    with open("springer_dataframe_safe.p", "rb") as f:
        df = pickle.load(f)

    embeddings_npz = np.load("doc_embeddings.npz")
    doc_embeddings = embeddings_npz["doc_embeddings"]

    with open("chunked_data.pkl", "rb") as f:
        chunked_docs, chunked_labels = pickle.load(f)

    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    return df, chunked_docs, chunked_labels, index

df, chunked_docs, chunked_labels, index = load_data_and_index()

# === Answer Generator ===
def generate_answer(prompt, model_choice):
    if model_choice == "Ollama":
        answer = ollama_generate(prompt, model="gemma3")
    elif model_choice == "Base FLAN-T5":
        answer = base_generator(prompt, max_length=200, do_sample=False)[0]['generated_text'].strip()
    elif model_choice == "Finetuned FLAN-T5":
        answer = finetuned_generator(prompt, max_length=200, do_sample=False)[0]['generated_text'].strip()
    else:
        answer = "Invalid model selected."
    return answer

# === RAG Query Handler ===
def rag_query(query, top_k=5, model_choice="Ollama"):
    query_clean = query.strip().lower()

    if query_clean in exact_qa_dict:
        exact_entry = exact_qa_dict[query_clean]
        context = exact_entry.get("context", "")

        prompt = (
            f"You are an expert academic assistant helping a student understand a topic. "
            f"Use the context to answer the question clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        generated_answer = generate_answer(prompt, model_choice)

        return {
            "query": query,
            "context": context,
            "base_answer": exact_entry["answer"],
            "finetuned_answer": generated_answer,
            "predicted_category": exact_entry.get("LCSH_Label", ""),
            "related_topics": [exact_entry.get("LCSH_Label", "")]
        }

    category_patterns = [
        r"suggest some books on (.+)", r"available books on (.+)",
        r"show books on (.+)", r"what books are available on (.+)"
    ]
    for pattern in category_patterns:
        match = re.match(pattern, query_clean)
        if match:
            label_req = match.group(1).strip().lower()
            matched_label = next((lbl for lbl in set(chunked_labels) if lbl.lower() == label_req), None)
            if matched_label:
                books = df[df['LCSH_Label'].str.lower() == matched_label.lower()]
                context = "\n".join(
                    [f"{row['title']} > {row['toc'].splitlines()[0] if row['toc'] else ''}" for _, row in books.iterrows()]
                )
                answer = f"{len(books)} book(s) found on '{matched_label}'."
                return {
                    "query": query, "context": context,
                    "base_answer": answer, "finetuned_answer": answer,
                    "predicted_category": matched_label, "related_topics": [matched_label]
                }

    label_matches = [lbl for lbl in set(chunked_labels) if lbl.lower() == query_clean]
    if label_matches:
        matched_label = label_matches[0]
        books = df[df['LCSH_Label'].str.lower() == matched_label.lower()]
        context = "\n".join(
            [f"{row['title']} > {row['toc'].splitlines()[0] if row['toc'] else ''}" for _, row in books.iterrows()]
        )
        answer = f"{len(books)} book(s) found in the '{matched_label}' category."
        return {
            "query": query,
            "context": context,
            "base_answer": answer,
            "finetuned_answer": answer,
            "predicted_category": matched_label,
            "related_topics": [matched_label]
        }

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    retrieved_chunks = [chunked_docs[i] for i in I[0]]
    retrieved_labels = [chunked_labels[i] for i in I[0]]

    scores = cross_encoder.predict([[query, doc] for doc in retrieved_chunks])
    reranked_indices = np.argsort(scores)[::-1]
    context_docs = [retrieved_chunks[i] for i in reranked_indices[:3]]
    context = "\n".join(context_docs)

    prompt = (
        f"You are an expert academic assistant helping a student understand a topic based on a book's table of contents. "
        f"Summarize only the relevant points from the context. Be concise, avoid generic phrases, and avoid repetition. "
        f"Using the context below, write a clear, complete sentence to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    base_answer = generate_answer(prompt, model_choice)
    predicted_category = Counter([retrieved_labels[i] for i in reranked_indices[:3]]).most_common(1)[0][0]

    all_keywords = list(set([label.strip() for label in chunked_labels]))
    keyword_embeddings = embedder.encode(all_keywords, convert_to_numpy=True)
    keyword_index = faiss.IndexFlatL2(keyword_embeddings.shape[1])
    keyword_index.add(keyword_embeddings)
    D_tag, I_tag = keyword_index.search(query_embedding, 5)
    related_topics = [all_keywords[i] for i in I_tag[0]]

    return {
        "query": query,
        "context": context,
        "base_answer": base_answer,
        "finetuned_answer": base_answer,
        "predicted_category": predicted_category,
        "related_topics": related_topics
    }

# === CSS Styling ===
st.markdown("""
<style>
html, body, .stApp { background-color: #121212; color: #ffffff; }
.stTextInput input { background-color: #1e1e1e; color: white; border: 2px solid #90caf9; border-radius: 10px; padding: 8px; }
.tag { background: linear-gradient(to right, #2193b0, #6dd5ed); color: white; padding: 6px 14px; border-radius: 20px; font-size: 14px; white-space: nowrap; }
.book-card { background: #1e1e1e; color: #ffffff; padding: 15px; margin-bottom: 10px; border-radius: 10px; box-shadow: 0 2px 8px rgba(255,255,255,0.05); }
.answer-box { background-color: #2c2c2c; padding: 12px; border-radius: 8px; }
.chat-history { background-color: #2e3b4e; padding: 10px; border-radius: 8px; margin-bottom: 12px; }
.category-scroll { display: flex; flex-wrap: wrap; gap: 8px; padding: 8px; overflow-y: auto; justify-content: flex-start; }
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.markdown("### 📘 Available Categories")
    category_html = "<div class='category-scroll'>"
    for cat in sorted(set(chunked_labels)):
        category_html += f"<span class='tag'>{cat}</span>"
    category_html += "</div>"
    st.markdown(category_html, unsafe_allow_html=True)

    st.markdown("### 🔁 Select Model")
    model_choice = st.radio("Model Preference", ["Ollama", "Base FLAN-T5", "Finetuned FLAN-T5"])

# === Main App ===
st.markdown("<h1 style='text-align:center; color:#90caf9;'>📖 Intelligent Library Assistant</h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_topics" not in st.session_state:
    st.session_state.last_topics = []

query = st.text_input("📝 Ask your question", placeholder="e.g., Suggest some books on Artificial Intelligence")

if query:
    with st.spinner("Thinking..."):
        result = rag_query(query, model_choice=model_choice)
        st.session_state.chat_history.append(result)
        st.session_state.last_topics = result["related_topics"]

        st.subheader("Response")
        st.markdown(f"#### ✨ Model Used: {model_choice}")

        if query.strip().lower() in exact_qa_dict:
            st.markdown("##### 🟩 Reference Answer (Ground Truth)")
            st.markdown(f"<div class='answer-box'>{exact_qa_dict[query.strip().lower()]['answer']}</div>", unsafe_allow_html=True)

            st.markdown("##### 🤖 Generated Answer (Model Output)")
            st.markdown(f"<div class='answer-box'>{result['finetuned_answer']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='answer-box'>{result['finetuned_answer']}</div>", unsafe_allow_html=True)

        st.subheader("📖 Suggested Books from Retrieved Context")
        matched_titles = [re.split(r'>', chunk)[0].strip() for chunk in result['context'].split('\n')]
        matched_books = df[df['title'].isin(matched_titles)].drop_duplicates(subset="title").reset_index(drop=True)

        if not matched_books.empty:
            if "book_page" not in st.session_state:
                st.session_state.book_page = 0

            BOOKS_PER_PAGE = 10
            total_books = len(matched_books)
            total_pages = (total_books + BOOKS_PER_PAGE - 1) // BOOKS_PER_PAGE
            start_idx = st.session_state.book_page * BOOKS_PER_PAGE
            end_idx = start_idx + BOOKS_PER_PAGE

            for _, row in matched_books.iloc[start_idx:end_idx].iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="book-card">
                        <strong>{row['title']}</strong><br>
                        <span style="font-size: 12px; color: grey;">ID: {row['idbook']}</span><br>
                    </div>
                    """, unsafe_allow_html=True)
                    if row['toc']:
                        toc_text = row['toc'].strip()

                        # Generate a summary using TOC
                        summary_prompt = (
                            f"You are a helpful assistant. Given the Table of Contents of a book, generate a concise summary "
                            f"highlighting what the book covers.\n\n"
                            f"Table of Contents:\n{toc_text}\n\nSummary:"
                        )

                        try:
                            summary = generate_answer(summary_prompt, model_choice=model_choice)
                        except Exception as e:
                            summary = f"Error generating summary: {e}"

                        # Show summary before TOC
                        st.markdown(f"<div class='answer-box'><strong>📝 Summary:</strong> {summary}</div>", unsafe_allow_html=True)

                        with st.expander("📑 Table of Contents"):
                            st.markdown(f"<pre>{toc_text}</pre>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.session_state.book_page > 0:
                    if st.button("⬅️ Previous"):
                        st.session_state.book_page -= 1
            with col3:
                if st.session_state.book_page < total_pages - 1:
                    if st.button("Next ➡️"):
                        st.session_state.book_page += 1
            with col2:
                st.markdown(f"<p style='text-align:center;'>📄 Page {st.session_state.book_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
        else:
            st.warning("No matching books found from the retrieved chunks.")

        if st.session_state.last_topics:
            st.markdown("#### 🔖 Related Topics")
            st.markdown(" ".join([f"<span class='tag'>{tag}</span>" for tag in st.session_state.last_topics]), unsafe_allow_html=True)

if st.session_state.chat_history:
    with st.expander("📜 Chat History"):
        for item in reversed(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-history">
                <strong>Q:</strong> {item['query']}<br>
                <strong>Answer:</strong> {item['finetuned_answer'][:100]}...
            </div>
            """, unsafe_allow_html=True)
