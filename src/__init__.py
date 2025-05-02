"""
RAG-Based Document Question Answering System
-------------------------------------------
This project implements a Retrieval-Augmented Generation (RAG) system for
document querying. It uses OpenAI embeddings, Pinecone vector database, and the
LangChain framework.

Requirements:
- Python 3.11+
- langchain
- pinecone-client
- openai
- tiktoken
- pypdf
- nltk
"""

__all__ = ["DocumentProcessor", "RAG_QA", "VectorStore"]
