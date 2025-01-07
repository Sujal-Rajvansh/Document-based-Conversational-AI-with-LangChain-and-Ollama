# 📄 Document-Based Conversational AI with LangChain and Ollama 🚀

This project demonstrates building a **document-based conversational AI system** using **LangChain**, **FAISS**, and **Ollama** with the **Llama 3.2 model**. The system processes documents like PDFs, embeds them into a vector database, and enables question answering with context from the document.

## 🌟 Key Features
- **Document Ingestion**: Load and process PDF documents for conversational AI workflows.  
- **Vector Database with FAISS**: Embed document chunks for fast and efficient retrieval.  
- **Ollama Integration**: Utilize the powerful Llama 3.2 model for text generation and question answering.  
- **LangChain Orchestration**: Combine document retrieval and AI responses into a seamless pipeline.

## 🛠️ Installation
1. Install the required libraries:
   ```bash
   pip install langchain langchain-community sentence-transformers faiss-gpu pypdf langchain_ollama
Set up Colab Xterm (if running in Colab):
bash
Copy code
pip install colab-xterm
Log in to Hugging Face:
python
Copy code
from huggingface_hub import login
login("your_huggingface_token")
🚀 Workflow
1️⃣ Load and Process the Document
Use PyPDFLoader to load a PDF and split it into chunks:

python
Copy code
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Load the document
loader = PyPDFLoader("/content/sample_document.pdf")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)
2️⃣ Create a Vector Store with FAISS
Embed the document chunks and store them in a FAISS vector database:

python
Copy code
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load HuggingFace embeddings
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Save and load the vector store
vectorstore.save_local("faiss_index_")
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = persisted_vectorstore.as_retriever()
3️⃣ Load Llama 3.2 Model with Ollama
Use Ollama to load the Llama 3.2 model for conversational AI:

python
Copy code
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")
response = llm.invoke("Tell me a joke")
print(response)
4️⃣ Retrieval-Based Conversational AI
Combine the Llama model with RetrievalQA from LangChain:

python
Copy code
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Interactive query loop
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print(result)
📚 Example Interaction
Input: What is this document about?
Output: This document appears to be a sample PDF document created for testing purposes. It covers Technology, Health, and Environment topics.

Input: What are its contents?
Output: The document's content includes "sample text on different topics," specifically Technology, Health, and Environment.

Input: Tell me a joke.
Output: What do you call a fake noodle? An impasta.

🎯 Customization Options
Change Embedding Model: Use another embedding model like all-MiniLM-L6-v2 for specific tasks.
Adjust Chunk Size: Modify chunk_size and chunk_overlap to process smaller or larger sections of the document.
Use Custom Models: Replace Llama 3.2 with other Ollama-compatible models for experimentation.
Fine-Tune Llama: Train Llama 3.2 with domain-specific data for tailored responses.
📦 Model and Database Persistence
Save Vector Store
python
Copy code
vectorstore.save_local("faiss_index_")
Load Vector Store
python
Copy code
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserializa
