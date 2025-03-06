import os
import fitz
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

DEBUG_MODE = False

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    """Extract selectable text from the PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_text_from_images(pdf_path):
    """Extracts text from images inside a PDF using OCR (Tesseract) and saves debug images."""
    doc = fitz.open(pdf_path)
    image_texts = []

    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_data = base_image["image"]
            img_bytes = io.BytesIO(img_data)
            img = Image.open(img_bytes)

            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                image_texts.append(f"Page {i + 1}, Image {img_index + 1} Text:\n{ocr_text}")

    return "\n".join(image_texts)

def prepare_vector_db(pdf_folder: str = "./data", db_folder: str = "./vectorstore"):
    """Processes PDFs, extracts text + OCR from images, and stores data in ChromaDB."""
    
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    if os.path.exists(db_folder) and len(os.listdir(db_folder)) > 0:
        print("Using existing VectorDB, updating with new files...")
        vectorstore = Chroma(persist_directory=db_folder, embedding_function=embedding)
    else:
        print("No existing VectorDB found. Creating a new one...")
        vectorstore = None

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDF files found in the specified folder!")

    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Processing: {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        ocr_text = extract_text_from_images(pdf_path)
        combined_text = text + "\n\n" + ocr_text if ocr_text else text

        if DEBUG_MODE:
            print("\n🔍 Extracted Text Sample (Before Storing in VectorDB):\n")
            print(combined_text[:2000])
            print("\n... [Truncated Output] ...\n")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.create_documents([combined_text])

        all_docs.extend(split_docs)

    if vectorstore:
        print("Updating ChromaDB with new documents...")
        vectorstore.add_documents(all_docs)
    else:
        print("Creating new ChromaDB...")
        vectorstore = Chroma.from_documents(all_docs, embedding, persist_directory=db_folder)

    print("ChromaDB is ready! You can now query your PDFs.")

if __name__ == "__main__":
    prepare_vector_db()
