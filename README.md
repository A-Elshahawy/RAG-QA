# RAG-Based Document Question Answering System

This repository implements a Retrieval-Augmented Generation (RAG) system for accurate document querying. The system utilizes OpenAI embeddings, Pinecone vector database, and the LangChain framework to provide precise answers to questions based on document content.

## Project Overview

The Document QA system combines vector embeddings and large language models to enable accurate question answering about document contents. The system follows these key steps:

1. **Document Processing** : PDF documents are loaded and split into manageable chunks
2. **Vector Embedding** : Document chunks are converted to vector embeddings and stored in Pinecone
3. **Retrieval** : Relevant document chunks are retrieved based on query similarity
4. **Generation** : An LLM generates precise answers based on the retrieved context

## Features

* PDF document loading and processing
* Intelligent chunk size management with token counting
* Semantic search using OpenAI embeddings
* Vector storage with Pinecone
* Context-aware prompting for accurate answers
* Source attribution for transparency

## Requirements

* Python 3.11+
* OpenAI API key
* Pinecone API key and environment

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/A-Elshahawy/rag-document-qa.git
   cd rag-document-qa
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

## Usage

### 1. Prepare Your Documents

Place your PDF documents in the `documents` folder. The system will process all PDFs in this directory.

### 2. Run the Document Processing Pipeline

```python
from rag_qa import DocumentProcessor, VectorStore

# Initialize document processor
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)

# Load and process documents
documents = processor.load_documents("./documents")
document_chunks = processor.split_documents(documents)

# Initialize vector store
vector_store = VectorStore(index_name="my-documents")

# Store embeddings
vectorstore = vector_store.store_embeddings(document_chunks)
```

### 3. Query Your Documents

```python
from rag_qa import RAG_QA

# Initialize RAG QA system
rag_qa = RAG_QA(vectorstore)

# Ask questions
question = "What are the key features of our product?"
response = rag_qa.answer_question(question)

# Display answer and sources
print(f"Answer: {response['answer']}")
print("Sources:")
for i, doc in enumerate(response["source_documents"][:2]):
    print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', 'Unknown')}")
```

### 4. Run the Complete Example

The main function demonstrates the complete pipeline:

```bash
python rag_qa.py
```

## Advanced Configuration

### Customizing Chunk Size

Adjust the document chunk size based on your document type:

```python
# For technical documents with complex information
processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

# For narrative documents where context is important
processor = DocumentProcessor(chunk_size=1500, chunk_overlap=300)
```

### Using Different LLM Models

Change the underlying LLM model:

```python
# Use GPT-4 for higher accuracy
rag_qa = RAG_QA(vectorstore, model_name="gpt-4", temperature=0)

# Use a more economical model
rag_qa = RAG_QA(vectorstore, model_name="gpt-3.5-turbo-16k", temperature=0)
```

### Custom Prompting

Customize the system prompt for different use cases:

```python
custom_template = """
You are a specialized assistant that answers questions about legal documents.
Always cite specific sections from the source document.

Context:
{context}

Question:
{question}

Please provide a legally precise answer based ONLY on the information in the context.
Include relevant citations to specific sections.
"""

rag_qa = RAG_QA(vectorstore, custom_prompt_template=custom_template)
```

## Architecture Diagram

```
┌────────────────┐    ┌──────────────────┐    ┌──────────────┐
│                │    │                  │    │              │
│  PDF Document  │───▶│  Document Chunks  │───▶│  Embeddings  
│                │    │                  │    │              │
└────────────────┘    └──────────────────┘    └──────┬───────┘
                                                     │
                                              	    🔽
┌────────────────┐    ┌──────────────────┐    ┌──────────────┐
│                │    │                  │    │              │
│    Answer      │◀───│  LLM Generation  │◀───│   Pinecone   
│                │    │                  │    │              │
└────────────────┘    └──────────────────┘    └──────────────┘
```

## Troubleshooting

### Connection Issues with Pinecone

If you encounter connection errors with Pinecone:

1. Verify your API key and environment variables
2. Check if you're using the correct environment region
3. Ensure your Pinecone service tier supports the index dimension (1536 for OpenAI embeddings)

### OpenAI API Rate Limits

If you encounter rate limit errors with OpenAI:

1. Implement exponential backoff retry logic
2. Consider batching embeddings requests
3. Upgrade your OpenAI plan for higher rate limits

## License

This project is licensed under the MIT License - see the LICENSE file for details.
