"""
RAG-Based Document Question Answering System
-------------------------------------------
This project implements a Retrieval-Augmented Generation (RAG) system for
document querying. It uses HuggingFaceBgeEmbeddings, Pinecone vector database, and the
LangChain framework.

Requirements:
nltk>=3.9.1
pypdf>=5.4.0
tqdm>=4.67.1
pinecone>=6.0.2
tiktoken>=0.9.0
langchain>=0.3.24
python-dotenv>=1.1.0
langchain-community>=0.3.23
sentence-transformers>=4.1.0
"""

__all__ = ["DocumentProcessor", "RAG_QA", "VectorStore"]
