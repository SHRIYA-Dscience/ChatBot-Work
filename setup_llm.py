import re
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# --- Step 1: Load Local LLM using Hugging Face Transformers ---
def load_llm_local(model_id="distilgpt2"):
    """Load the local LLM using Hugging Face Transformers pipeline."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        tokenizer.pad_token = tokenizer.eos_token  # Avoid warning

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,
            top_p=1.0,
            top_k=50,
            no_repeat_ngram_size=2
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Step 2: Custom Prompt ---
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and knowledgeable assistant. Your goal is to provide clear and concise answers based solely on the provided context.

Context:
{context}

User Question:
{question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    """Set the custom prompt template."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# --- Step 3: Load Vectorstore (FAISS) ---
def load_vectorstore():
    """Load the FAISS vectorstore."""
    try:
        print("Loading FAISS vectorstore...")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

# --- Step 4: Build QA Chain ---
def build_qa_chain():
    """Build the QA chain with the vectorstore and LLM."""
    db = load_vectorstore()
    if not db:
        print("Failed to load vectorstore.")
        return None

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm_local(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

# --- Step 5: Clean Repetition ---
def clean_repetition(text):
    """Remove repeated words or short phrases."""
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text)  # repeated words
    text = re.sub(r'\b((?:\w+\s+){1,4}\w+)(?:\s+\1){2,}', r'\1', text)  # repeated phrases
    return text

# --- Main Execution ---
def main():
    """Main execution for handling user queries."""
    qa_chain = build_qa_chain()
    if not qa_chain:
        print("Failed to build QA chain.")
        return

    while True:
        user_query = input("Write Query Here (or type 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            break
        try:
            response = qa_chain.invoke({'query': user_query})  # ← FIXED here
            cleaned_response = clean_repetition(response['result'])

            print(f"\nQuery: {user_query}")
            print(f"Result: {cleaned_response}\n")
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
