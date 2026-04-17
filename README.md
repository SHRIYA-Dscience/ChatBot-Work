# AI-Powered Document Q&A Chatbot
An intelligent chatbot that answers user queries based on PDF documents using Retrieval-Augmented Generation (RAG). This project combines LangChain, FAISS, Hugging Face Transformers, and Streamlit to deliver accurate, context-aware responses through a user-friendly chat interface.

## Features
1. PDF based Question Answering
2. RAG Architecture (Retrieval + Generation)
3. Semantic Search using FAISS
4. Local LLM (Hugging Face Transformers)
5. nteractive Chat UI (Streamlit)
6.  Optimized with Caching
## Tech Stack
- Programming Language: Python
- Frameworks/Libraries:
-LangChain
- FAISS
- Hugging Face Transformers
- Streamlit
- Embedding Model: sentence-transformers/paraphrase-MiniLM-L6-v2
- LLM: distilgpt2 (can be upgraded)
---
## Installation 
1. **Clone the Repository**
```
git clone https://github.com/SHRIYA-Dscience/ChatBot-Work.git
cd ChatBot-Work
```
2. **Create Virtual Environment**
```
python -m venv venv

# Mac/Linux
source venv/bin/activate  

# Windows
venv\Scripts\activate
```
3. **Install Dependencies**
```
pip install -r requirements.txt
```
---
