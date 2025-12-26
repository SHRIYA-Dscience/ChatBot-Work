import os
import re
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
LLM_MODEL_ID = "distilgpt2"

# --- Prompt Template ---
CUSTOM_PROMPT_TEMPLATE =  """
You are a helpful and knowledgeable assistant. Your goal is to provide clear and concise answers based solely on the provided context.

Context:
{context}

User Question:
{question}

Answer:
"""

# --- Utilities ---
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def clean_repetition(text):
    text = re.sub(r'(\b\w+\b)(?:\s+\1){3,}', r'\1', text).strip()
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text)
    return text

# --- Streamlit Page Config & Styling ---
st.set_page_config(page_title="📚 AI Q&A Chatbot", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        .reportview-container { background-color: #f9f9f9; }
        .sidebar .sidebar-content { background-color: #ffffff; }
        .stChatMessage { border-radius: 0.5rem; padding: 0.75rem; }
        .stChatMessage.user { background-color: #e0f7fa; }
        .stChatMessage.assistant { background-color: #fff9c4; }
    </style>
""", unsafe_allow_html=True)

# --- Loaders with Caching ---
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"FAISS index not found at '{DB_FAISS_PATH}'. Please generate embeddings first.")
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm_local(model_id=LLM_MODEL_ID):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,
            top_p=1.0,
            top_k=50,
            no_repeat_ngram_size=2,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        return None

@st.cache_resource
def build_qa_chain():
    db = load_vectorstore()
    llm = load_llm_local()
    if not db or not llm:
        return None
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose LLM", ["distilgpt2", "gpt2-medium"], index=0)
reset_btn = st.sidebar.button("🔄 Reset Chat")

# --- Main ---
def main():
    st.title("📚 AI Q&A Chatbot")
    st.write("Ask coding-related questions based on your PDF corpus.")

    if reset_btn:
        st.session_state.clear()
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Your question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🤖 Thinking..."):
            qa_chain = build_qa_chain()
            if qa_chain:
                result = qa_chain.invoke({"query": prompt})
                answer = clean_repetition(result.get("result", ""))
            else:
                answer = "⚠️ Sorry, something went wrong loading the QA chain."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()
