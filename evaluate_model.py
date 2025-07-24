import json
import numpy as np
import evaluate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import faiss
import requests
import time
import nltk
nltk.download('punkt')

# --- Streamlit caching for models ---
import streamlit as st

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    base_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    finetuned_generator = pipeline("text2text-generation", model="./finetuned-flan-t5")
    return embedder, cross_encoder, base_generator, finetuned_generator

embedder, cross_encoder, base_generator, finetuned_generator = load_models()

# --- Load QA Data and split into train / eval ---
with open("diverse_qa_pairs.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

train_data, eval_data = train_test_split(qa_data, test_size=0.2, random_state=42)

# --- Build exact QA dictionary ---
exact_qa_dict = {
    entry["question"].strip().lower(): entry
    for entry in train_data
    if isinstance(entry.get("question", ""), str)
}

# --- Build FAISS index ---
corpus = [entry["context"] for entry in train_data]
corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True)
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

def retrieve_context(query, top_k=5):
    query_embedding = embedder.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    return [corpus[i] for i in I[0]]

def safe_generate(pipeline_obj, prompt, max_length=200, retries=3):
    for _ in range(retries):
        try:
            output = pipeline_obj(prompt, max_length=max_length, do_sample=False)
            if output and 'generated_text' in output[0]:
                text = output[0]['generated_text'].strip()
                if text:
                    return text
        except Exception:
            continue
    return "[Generation Failed]"

def ollama_generate(prompt, model="gemma3", retries=3):
    for _ in range(retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            if response.ok:
                answer = response.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception:
            time.sleep(1)
            continue
    return "[Ollama Generation Failed]"

def generate_answer(prompt, model_choice="finetuned"):
    if model_choice == "finetuned":
        return safe_generate(finetuned_generator, prompt)
    elif model_choice == "base":
        return safe_generate(base_generator, prompt)
    elif model_choice == "ollama":
        return ollama_generate(prompt, model="gemma3")
    else:
        return "[Unknown model response]"

def rag_query(query, top_k=5, force_generate=True):
    query_clean = query.strip().lower()

    if not force_generate and query_clean in exact_qa_dict:
        exact_entry = exact_qa_dict[query_clean]
        return {
            "query": query,
            "context": exact_entry.get("context", ""),
            "base_answer": exact_entry.get("answer", "N/A"),
            "finetuned_answer": exact_entry.get("answer", "N/A"),
            "ollama_answer": exact_entry.get("answer", "N/A"),
            "predicted_category": exact_entry.get("LCSH_Label", ""),
            "related_topics": [exact_entry.get("LCSH_Label", "")]
        }

    if query_clean in exact_qa_dict:
        context = exact_qa_dict[query_clean].get("context", "")
    else:
        retrieved_contexts = retrieve_context(query, top_k=top_k)
        context = "\n".join(retrieved_contexts)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    finetuned_answer = generate_answer(prompt, model_choice="finetuned")
    ollama_answer = generate_answer(prompt, model_choice="ollama")

    base_answer = exact_qa_dict.get(query_clean, {}).get("answer", "").strip()
    if not base_answer or base_answer.lower() == "n/a":
        base_answer = generate_answer(prompt, model_choice="base")

    return {
        "query": query,
        "context": context,
        "base_answer": base_answer,
        "finetuned_answer": finetuned_answer,
        "ollama_answer": ollama_answer
    }

# --- Load only BERTScore ---
bertscore = evaluate.load("bertscore")

# --- Define answer sources ---
answer_sources = ["base_answer", "finetuned_answer", "ollama_answer"]
results = {source: {"predictions": [], "references": []} for source in answer_sources}
references = []

print("\n🔍 Evaluating All Answer Sources (force_generate=True)...")
for i, sample in enumerate(tqdm(eval_data)):
    question = sample.get("question", "").strip()
    gold_answer = sample.get("answer", "").strip()
    if not question or not gold_answer:
        continue

    result = rag_query(question, top_k=5, force_generate=True)
    references.append(gold_answer)
    print(f"\n🟡 Question {i+1}: {question}")
    print(f"✅ Reference Answer: {gold_answer}")
    for source in answer_sources:
        pred = result.get(source, "")
        if not isinstance(pred, str) or not pred.strip():
            pred = "No answer generated"
        results[source]["predictions"].append(pred.strip())
        results[source]["references"].append(gold_answer)
        print(f"🔵 {source}: {pred.strip()}")

# --- Compute and print only BERTScore ---
for source in answer_sources:
    print(f"\n📊 Evaluation Results for: {source}")
    preds = results[source]["predictions"]
    refs = results[source]["references"]
    bert = bertscore.compute(predictions=preds, references=refs, lang="en")
    bert_f1 = sum(bert["f1"]) / len(bert["f1"]) if len(bert["f1"]) > 0 else 0.0
    print(f"BERTScore F1:  {bert_f1:.4f}")
