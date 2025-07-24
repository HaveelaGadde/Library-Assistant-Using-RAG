import pandas as pd
import numpy as np
import faiss
import re
import os
import pickle
from sentence_transformers import SentenceTransformer

print("🔄 Loading model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("📖 Loading dataframe...")
with open("springer_dataframe_safe.p", "rb") as f:
    df = pickle.load(f)

chunked_docs, chunked_labels = [], []

print("📚 Performing hierarchical chunking...")

# Hierarchical chunking logic
for _, row in df.iterrows():
    title, toc, label = row['title'], row['toc'], row['LCSH_Label']
    
    # Skip invalid TOC entries
    if pd.isna(toc) or not isinstance(toc, str) or len(toc.strip()) == 0:
        continue

    raw_sections = re.split(r'[/\n\r]+', toc)
    hierarchy_stack = []

    for section in raw_sections:
        section = section.strip()
        if not section or len(section) < 5:
            continue

        match = re.match(r'^(\d+(\.\d+)?)(-|:)?\s(.+)', section)
        if match:
            numbering = match.group(1)
            content = match.group(4).strip()
            level = numbering.count('.')
            hierarchy_stack = hierarchy_stack[:level]
            hierarchy_stack.append(content)
            hierarchy_path = f"{title} > " + " > ".join(hierarchy_stack)
        else:
            hierarchy_path = f"{title} > " + " > ".join(hierarchy_stack + [section]) if hierarchy_stack else f"{title} > {section}"

        chunked_docs.append(hierarchy_path)
        chunked_labels.append(label)

print(f"🔢 Total Hierarchical Chunks: {len(chunked_docs)}")

print("🔍 Encoding documents (this may take 30+ minutes on CPU)...")
doc_embeddings = embedder.encode(chunked_docs, convert_to_numpy=True, show_progress_bar=True)

print("💾 Saving to disk as compressed .npz...")
np.savez_compressed("doc_embeddings.npz", doc_embeddings=doc_embeddings)

with open("chunked_data.pkl", "wb") as f:
    pickle.dump((chunked_docs, chunked_labels), f)

print("✅ Done! Now restart your Streamlit app.")
