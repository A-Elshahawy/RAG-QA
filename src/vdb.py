import os
from typing import List

import pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Pinecone


class VectorStore:
    """Manages vector embeddings and storage."""

    def __init__(self, index_name="my-documents"):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.index_name = index_name

        # Initialize Pinecone client
        self.pinecone_client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Create index if it doesn't exist (MiniLM-L6-v2 uses 384 dimensions)
        print("\n\n  Pinecone idxes", self.pinecone_client.list_indexes().names())
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_ENV"),
                ),
            )
        # Get the index object
        self.index = self.pinecone_client.Index(self.index_name)

    def store_embeddings(self, document_chunks: List) -> Pinecone:
        """Store document chunks as embeddings in Pinecone."""
        vectorstore = Pinecone.from_documents(
            documents=document_chunks, embedding=self.embeddings, index=self.index
        )
        return vectorstore

    def load_vectorstore(self) -> Pinecone:
        """Load existing vector store from Pinecone."""
        return Pinecone.from_existing_index(embedding=self.embeddings, index=self.index)
