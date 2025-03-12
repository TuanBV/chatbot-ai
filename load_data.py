from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 🔹 Load tài liệu PDF
loader = PyPDFLoader("data.pdf")  # Đổi thành file của bạn
documents = loader.load()

# 🔹 Chia nhỏ văn bản
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 🔹 Tạo vector embeddings bằng SentenceTransformers
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Lưu dữ liệu vào FAISS
vector_store = FAISS.from_documents(texts, embedding_model)
vector_store.save_local("faiss_index")

print("✅ FAISS database đã được tạo và lưu!")
