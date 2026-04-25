---
title: RAG APP
emoji: 📊
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: Production-Grade RAG Chatbot
---
 LUP-Retriever-v1
**Production-Grade Document Intelligence at LPU Speed**

GroqFlow is a Retrieval-Augmented Generation (RAG) chatbot designed for high-performance enterprise search. It leverages the Groq LPU™ Inference Engine to provide near-instant responses from your private PDF and TXT datasets.



## ⚡ Key Features
- **Ultra-Fast Inference:** Powered by Groq LPU™ for Llama 3.3 / Mixtral models.
- **2026 Modular Architecture:** Fully compliant with LangChain v0.3+ and `langchain-classic`.
- **Local Vector Storage:** FAISS-based local indexing (zero-cost, high-speed).
- **Elegant Interface:** Gradio 6.0 "Discovery Hub" with streaming responses and semantic lookup.
- **Production Ready:** Optimized for Hugging Face Spaces (CPU-Basic).

## 🛠️ Tech Stack
- **LLM:** Groq (Llama-3.3-70b-versatile)
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
- **UI:** Gradio 6.0
- **Framework:** LangChain 2026 Modular Ecosystem

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Groq API Key ([Get it here](https://console.groq.com/))

### 2. Installation
```bash
pip install -r requirements.txt
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
