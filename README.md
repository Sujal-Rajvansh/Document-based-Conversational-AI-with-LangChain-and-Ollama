# 📄 Document-Based Conversational AI with LangChain and Ollama 🚀

This project demonstrates how to build a **document-based conversational AI system** using **LangChain**, **FAISS**, and **Ollama** integrated with **Llama 3.2 model**. It enables PDF document ingestion, embedding creation, vector storage, and retrieval-augmented question answering. The system is ideal for generating context-aware answers or interacting with structured content.

---

## 🌟 Key Features
- **Llama 3.2 Integration**: Powerful text generation model integrated using Ollama.
- **PDF Document Processing**: Load, split, and embed documents into FAISS for retrieval.
- **FAISS Vector Store**: Efficient vector storage and retrieval for contextual queries.
- **Interactive Text Generation**: Answer queries or generate text directly from documents.
- **LangChain Orchestration**: Combines document retrieval with language model reasoning.

---

## 🛠️ Installation

### Install Required Libraries
Run the following commands to set up the environment:
```bash
pip install langchain langchain-community sentence-transformers faiss-gpu pypdf langchain_ollama
Additional Setup
Install Colab Xterm (if running in Colab):
bash
Copy code
pip install colab-xterm
Log in to Hugging Face:
python
Copy code
from huggingface_hub import login
login("your_huggingface_token")
🚀 Workflow
1️⃣ Load and Process PDF Document
Use PyPDFLoader to load a PDF and split it into manageable chunks:

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
2️⃣ Create a FAISS Vector Store
Embed the document chunks and store them in FAISS for efficient retrieval:

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

# Save vector store locally
vectorstore.save_local("faiss_index_")

# Load vector store
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = persisted_vectorstore.as_retriever()
3️⃣ Load Llama Model Using Ollama
Use Ollama to load and interact with the Llama 3.2 model:

python
Copy code
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")
response = llm.invoke("Tell me a joke")
print(response)
4️⃣ Retrieval-Based Conversational AI
Combine the Llama model with RetrievalQA for context-aware question answering:

python
Copy code
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Interactive QA loop
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print(result)
📚 Example Interaction
Input: What is this document about?
Output: This document appears to be a sample PDF created for testing purposes, specifically for testing document reading and indexing functions.

Input: What are its contents?
Output: The document's content includes topics such as Technology, Health, and Environment.

Input: Tell me a joke.
Output: Here's one: What do you call a fake noodle? An impasta.

🎯 Customization Options
Change Embedding Model: Replace sentence-transformers/all-mpnet-base-v2 with another Hugging Face embedding model for better performance on specific tasks.
Modify Chunk Sizes: Adjust chunk_size and chunk_overlap in the text splitter for optimal document handling.
Fine-Tune Llama: Train Llama 3.2 with domain-specific data for tailored responses.
Add Custom Chains: Extend LangChain workflows to integrate advanced reasoning or data augmentation.
📦 Model and Database Persistence
Save Vector Store and Model
python
Copy code
# Save FAISS vector store
vectorstore.save_local("faiss_index_")

# Save fine-tuned model (if applicable)
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
Load Saved Assets
python
Copy code
# Load FAISS vector store
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Load fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/fine_tuned_model")
