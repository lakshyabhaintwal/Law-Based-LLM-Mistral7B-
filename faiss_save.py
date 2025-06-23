from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import json

# === Load dataset from Hugging Face ===
dataset = load_dataset("viber1/indian-law-dataset", split="train")

# === Prepare RAG docs ===
rag_docs = [
    f"Q: {item['Instruction']}\nA: {item['Response']}"
    for item in dataset
]

# === Save docs to file ===
with open("rag_docs.json", "w") as f:
    json.dump(rag_docs, f)

# === Embed documents ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(rag_docs, convert_to_numpy=True, show_progress_bar=True)

# === Build and save FAISS index ===
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "lawbot_index.faiss")

print("✅ FAISS index and RAG docs saved successfully.")
