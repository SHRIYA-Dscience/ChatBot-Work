import os
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# --- Load environment variables ---
load_dotenv(find_dotenv())

# --- Configuration ---
DATA_PATH = "Data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)

# --- Step 1: Load PDF Files with Metadata ---
def load_pdf_files(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory '{data_path}' does not exist.")

    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        use_multithreading=True,
    )
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF files found in the directory.")

    logging.info(f"Loaded {len(documents)} documents from {data_path}")
    return documents

# --- Step 2: Create Chunks with Metadata ---
def create_chunks(documents, chunk_size=500, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

# --- Step 3: Load Embedding Model ---
def load_embeddings(model_name=EMBEDDING_MODEL):
    logging.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# --- Step 4: Create and Save FAISS Index ---
def store_faiss_index(chunks, embeddings, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.info("Creating FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)
    logging.info(f"FAISS index saved at: {save_path}")

# --- Main Execution ---
def main():
    try:
        docs = load_pdf_files(DATA_PATH)
        chunks = create_chunks(docs)
        embeddings = load_embeddings()
        store_faiss_index(chunks, embeddings, DB_FAISS_PATH)
        logging.info("Vector store ready for chatbot queries.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
