# custom-chatbot-exploration

Learning to build a multimodal AI chatbot that can process text-based and image-based PDFs using OCR and vector search.  

## Quick Start  

- Add PDFs: Place your PDF files in the `data/` folder.  
- Install Dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
- Prepare VectorDB: Process PDFs and store embeddings.  
   ```bash
   python prepare_vectordb.py
   ```
- Run Chatbot: Launch the chatbot UI using Streamlit.  
   ```bash
   streamlit run chatbot_ui.py
   ```

## Key Features  

- Text & Image Processing: Extracts text from selectable and scanned PDFs.  
- Hybrid Search: Uses ChromaDB for fast, efficient document retrieval.  
- Tesseract OCR: Extracts text from figures and complex layouts.  
- Modular Design[TBD]: Easily swap out LLMs, vector stores, or embeddings.  

## Testing Setup  

The chatbot has been tested on two types of PDFs:  

1. Full-text PDFs => Standard, searchable documents.  
2. Multimodal PDFs (Text + Images) => Tested using Yu-Gi-Oh! rulebook: [Official PDF](https://img.yugioh-card.com/en/downloads/rulebook/SD_RuleBook_EN_10.pdf) => [TBD] Evaluate the results accuracy  

## Technical Details  

- Document Processing: PyMuPDF (`fitz`), `pytesseract` (OCR)  
- VectorDB Storage: ChromaDB  
- Embeddings: OpenAI's `text-embedding-ada-002`  
- Frontend: Streamlit  


## Next Steps
- Understand how to support complex layouts (tables, multi-column text, figures with explanation)  
- Explore Fine-tuning responses with user feedback  
- Integrating different embedding models for better retrieval
- Try a custom LLM model
