import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

load_dotenv()

def prepare_vector_db(pdf_folder: str = "./data", db_folder: str = "./vectorstore"):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(db_folder) and len(os.listdir(db_folder)) > 0:
        print("Using existing VectorDB, updating with new files...")
        vectorstore = Chroma(persist_directory=db_folder, embedding_function=embedding)
    else:
        print("No existing VectorDB found. Creating a new one...")
        vectorstore = None

    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError("⚠️ No PDF files found in the specified folder!")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)

    if vectorstore:
        print("Updating ChromaDB with new documents...")
        vectorstore.add_documents(split_docs)
    else:
        print("Creating new ChromaDB...")
        vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory=db_folder)

    print("ChromaDB is ready!")

if __name__ == "__main__":
    prepare_vector_db()
