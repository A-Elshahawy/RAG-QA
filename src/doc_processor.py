from typing import List

import nltk
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

# Download necessary NLTK data
nltk.download("punkt")


class DocumentProcessor:
    """Handles document loading and chunking."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def load_documents(self, directory_path: str) -> List:
        """Load PDF documents from a directory."""
        loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        document_chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(document_chunks)} chunks")
        return document_chunks

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        tokens = tokenizer.encode(text)
        return len(tokens)
