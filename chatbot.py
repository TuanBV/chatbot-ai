from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 🔹 Load FAISS database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model)

# 🔹 Dùng mô hình Ollama (Mistral, Gemma, LLaMA...)
llm = Ollama(model="mistral")  # Bạn có thể thay "mistral" bằng "gemma", "phi", v.v.

# 🔹 Tạo chatbot RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff",
)

# 🔹 Vòng lặp chat
print("✅ RAG Chatbot đã sẵn sàng! Nhập 'exit' để thoát.")
while True:
    query = input("Bạn: ")
    if query.lower() == "exit":
        print("Chatbot: Tạm biệt!")
        break
    response = rag_chain.run(query)
    print(f"Chatbot: {response}")
