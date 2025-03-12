# Create environment

python -m venv env

# install package by requirements.txt

pip install -r requirements.txt

# Create FAISS database

python load_data.py

# Search answer in FAISS database and return response by Ollama

python chatbot.py

# Show Web application with Streamlit

streamlit run app.py
