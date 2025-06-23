import faiss
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# === Config ===
model_path = "Mistral_merged"                  # ✅ your local fine-tuned model
index_path = "lawbot_index.faiss"              # ✅ saved FAISS index
rag_docs_path = "rag_docs.json"                # ✅ source docs
embed_model_name = "all-MiniLM-L6-v2"          # ✅ pretrained embedder

# === Load tokenizer and model (no bitsandbytes needed) ===
tokenizer = AutoTokenizer.from_pretrained(model_path)

'''model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",                         # ✅ maps to GPU automatically
    torch_dtype=torch.float16,                 # ✅ safe on RTX 4070
    low_cpu_mem_usage=True                     # ✅ helps avoid RAM spikes
)'''
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # safer on Windows
    device_map={"": "cpu"},     # force CPU
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)
model.eval()

# === Load embedding model ===
embed_model = SentenceTransformer(embed_model_name)

# === Load FAISS index ===
if not os.path.exists(index_path):
    raise FileNotFoundError("❌ FAISS index not found! Run index builder first.")
index = faiss.read_index(index_path)

# === Load RAG documents ===
if not os.path.exists(rag_docs_path):
    raise FileNotFoundError("❌ RAG doc file not found. Please check rag_docs.json.")
with open(rag_docs_path, "r", encoding="utf-8") as f:
    rag_docs = json.load(f)

# === RAG context retriever ===
def retrieve_context(query, top_k=3):
    query_vec = embed_model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [rag_docs[i] for i in I[0]]

# === Chat interface with LLM and RAG ===
def rag_chat(query):
    context = retrieve_context(query)
    prompt = (
        "<s>[INST] You are a legal assistant. Use the following context to answer the query.\n\n"
        + "\n\n".join(context) +
        f"\n\nQuestion: {query}\nAnswer: [/INST]"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
