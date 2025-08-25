# PDF Q&A with Streamlit (RAG)

This project is a simple **Retrieval-Augmented Generation (RAG)** app built with **Streamlit** and **LangChain**.  
Users can upload a PDF, ask questions, and get AI-powered answers based on the document.

---

## 🚀 Features
- Upload PDF files
- Extract text from PDF
- Store embeddings with FAISS
- Ask questions about the content
- Uses Hugging Face models for embeddings and responses
- Includes a **notebook** (`basic-rag-system.ipynb`) with a step-by-step explanation of the RAG pipeline

---

## 📓 Jupyter Notebook: `basic-rag-system.ipynb`

This notebook was developed on **Kaggle** and demonstrates the complete workflow of a **basic RAG system**.  

### 🔹 Introduction
- Explains the motivation behind Retrieval-Augmented Generation (RAG).  
- Shows how language models can be combined with vector search to answer document-based questions.  
- Uses **Mistral** (via Hugging Face), **Sentence Transformers**, and **FAISS**.  

### 🔹 Steps Covered
1. **Model Loading**  
   - Loads Mistral (`mistralai/Mistral-Nemo-Instruct-2407`) from Hugging Face.  
   - Prepares tokenizer & inference pipeline.  

2. **PDF Processing**  
   - Extracts text from PDF using `PyPDF2`.  
   - Splits text into overlapping chunks.  

3. **Embeddings & Indexing**  
   - Embeds text chunks with `sentence-transformers/all-MiniLM-L6-v2`.  
   - Stores embeddings inside a **FAISS vector database**.  

4. **Search & Retrieval**  
   - Queries FAISS index with user questions.  
   - Retrieves top-k most relevant chunks.  

5. **LLM Answer Generation**  
   - Feeds retrieved chunks + question into Mistral model.  
   - Generates detailed answers.  

### 🔹 Conclusion
- Demonstrates how **RAG improves factual accuracy** compared to plain LLM prompts.  
- Shows a complete **end-to-end pipeline**: PDF → Embeddings → Vector Search → LLM Answer.  
- Serves as a learning foundation before deploying the **Streamlit App**.  

---

## 📦 Installation
Clone the repo:
```bash
git clone https://github.com/yourusername/pdf-rag.git
cd pdf-rag
```

Create virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 🔑 Setup Hugging Face Token
You need a Hugging Face API token. Create an account at [huggingface.co](https://huggingface.co), then add your token inside `.env` file:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Or set it in your script:
```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"
```

### ▶️ Run the App
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```
pdf-rag/
├── app.py                   # Streamlit app
├── basic-rag-system.ipynb   # Notebook with step-by-step RAG pipeline
├── requirements.txt         # Dependencies for app and notebook
├── README.md               # Project documentation
└── .env.example            # Example environment file
```

---

## 🙌 Acknowledgments
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [Kaggle](https://www.kaggle.com/) for hosting the notebook

---

## ✨ Connet with me
- ![Portfolio](https://sites.google.com/view/abdelrahman-eldaba110)
- ![LinkedIn](https://www.linkedin.com/in/abdelrahmaneldaba)
- ![Kaggle](https://www.kaggle.com/abdelrahmanahmed110)