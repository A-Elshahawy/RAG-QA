from src.doc_processor import DocumentProcessor
from src.rag_qa import RAG_QA
from src.vdb import VectorStore


def main():
    """Main function to demonstrate the RAG system."""
    # Initialize document processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)

    # Load and process documents
    documents = processor.load_documents("./documents")
    document_chunks = processor.split_documents(documents)

    # Initialize vector store
    vector_store = VectorStore(index_name="my-documents")

    # Store embeddings
    vectorstore = vector_store.store_embeddings(document_chunks)

    # Initialize RAG QA system
    rag_qa = RAG_QA(vectorstore)

    # Example questions
    # questions = [
    #     "What are the key features of our product?",
    #     "How does our pricing model work?",
    #     "What technical specifications should I know about?",
    # ]
    questions = [
        "What is DistilBERT, and how does it differ from BERT",
        "What are the key advantages of DistilBERT compared to traditional BERT models",
        "How was DistilBERT trained and optimized for performance",
        "What are the primary use cases for DistilBERT in NLP applications",
    ]

    # Answer questions
    for question in questions:
        print(f"\nQuestion: {question}")
        response = rag_qa.answer_question(question)
        print(f"Answer: {response['answer']}")
        print("Sources:")
        for i, doc in enumerate(response["source_documents"][:2]):
            print(
                f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')},\
                    page {doc.metadata.get('page', 'Unknown')}"
            )


if __name__ == "__main__":
    main()
