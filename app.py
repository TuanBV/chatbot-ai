import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 🔹 Load FAISS database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization = True)

# 🔹 Dùng mô hình Ollama (Mistral, Gemma, LLaMA...)
llm = Ollama(model="mistral")

# 🔹 Tạo chatbot RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
)

# 🔹 Giao diện Streamlit
st.set_page_config(page_title="RAG Chatbot với Ollama", layout="centered")
st.title("🤖 RAG Chatbot - Hỏi Đáp từ Tài Liệu 📄")

query = st.text_input("Nhập câu hỏi của bạn:", "")

if st.button("Gửi"):
    if query:
        print(query)
        response = rag_chain.invoke({"query": query})
        print(response)
        st.write(f"**Chatbot:** {response}")
    else:
        st.warning("Vui lòng nhập câu hỏi!")

# Chạy bằng lệnh: streamlit run app.py
