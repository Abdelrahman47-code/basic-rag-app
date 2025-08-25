import os
import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="PDF Q&A App",
    page_icon="📘",
    layout="wide"
)

# -----------------------------
# Sidebar (Author Info & Navigation)
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=80)
    st.markdown("## 👨‍💻 Built by")
    st.markdown("**Abdelrahman Eldaba**")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/abdelrahmaneldaba/)")
    st.markdown("---")
    st.info("Upload a PDF and ask questions!\n\nThe app uses **RAG (Retrieval-Augmented Generation)** with **HuggingFace models**.")

# -----------------------------
# Main UI
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>📘 PDF Q&A App</h1>
    <p style='text-align: center; color: gray;'>Upload a PDF and ask any question about its content!</p>
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("⏳ Processing your PDF..."):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()

        # HuggingFace pipeline (local inference, no API call)
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

        # Wrap pipeline with LangChain
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Retrieval-QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDF indexed successfully!")

        # -----------------------------
        # Ask questions
        # -----------------------------
        st.subheader("💬 Ask your question")
        query = st.text_input("Type your question below:")
        submit = st.button("🔎 Get Answer")

        if submit and query:
            with st.spinner("🤔 Thinking..."):
                answer = qa_chain.run(query)

                # If no meaningful answer is returned
                if not answer or answer.strip() in ["", "I don't know", "No answer found."]:
                    st.warning("⚠️ Sorry, I couldn’t find an answer in the PDF. Try rephrasing your question.")
                else:
                    st.markdown("### ✅ Answer")
                    st.write(answer)

                    # Save Q&A to session for history
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append((query, answer))

        # -----------------------------
        # Q&A History
        # -----------------------------
        if "history" in st.session_state and st.session_state["history"]:
            st.markdown("### 📝 Your Previous Questions")
            for q, a in st.session_state["history"]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")
