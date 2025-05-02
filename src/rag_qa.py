from typing import Any

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class RAG_QA:
    """RAG-based question answering system."""

    def __init__(self, vectorstore, model_name="gpt-3.5-turbo", temperature=0):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        # Define a custom prompt template
        template = """
        You are an assistant that answers questions based on the provided context.

        Context:
        {context}

        Question:
        {question}

        Please provide a detailed answer based ONLY on the information in the context.
        If the context doesn't contain the relevant information,
        say "I don't have enough information to answer this question."
        """

        self.prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        # Create the RAG chain
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the question-answering chain."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},  # Retrieve top 4 most similar documents
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def answer_question(self, question: str) -> dict[str, Any]:
        """Answer a question using the RAG system."""
        response = self.qa_chain({"query": question})

        return {
            "question": question,
            "answer": response["result"],
            "source_documents": response["source_documents"],
        }
