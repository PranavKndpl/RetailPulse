import os
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class LLMEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment")

        # charts & visuals - fast
        self.fast_llm = ChatGroq(
            api_key=SecretStr(api_key),
            model="llama-3.1-8b-instant",
            temperature=0.2
        )

        # forecast interpretation - smart
        self.smart_llm = ChatGroq(
            api_key=SecretStr(api_key),
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )

        #RAG for SOPs
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_db = None

    def update_knowledge_base(self, text_content: str):
        if not text_content.strip():
            self.vector_db = None
            return "[INFO] Knowledge Base cleared."

        doc = Document(
            page_content=text_content,
            metadata={"source": "user_rules"}
        )

        self.vector_db = Chroma.from_documents(
            documents=[doc],
            embedding=self.embedding_function
        )
        return "[SUCCESS] SOPs updated."

    def clear_knowledge_base(self):
        self.vector_db = None
        return "Memory cleared."

    def _get_rag_context(self, query: str) -> str:
        if self.vector_db:
            docs = self.vector_db.similarity_search(query, k=1)
            if docs:
                return docs[0].page_content
        return "No specific SOPs provided."


    def analyze_visuals(self, data_summary: str) -> str:
        rag_context = self._get_rag_context("visual interpretation")

        prompt = ChatPromptTemplate.from_template(
            """
            You are a neutral data analyst explaining observed trends.

            RULES:
            - Do not give advice.
            - Do not mention model performance.
            - Do not invent numbers.
            - Focus only on observable patterns.
            - Keep language concise and professional.

            SOP CONTEXT:
            {rag_context}

            DATA SUMMARY:
            {data_summary}

            OUTPUT:
            Provide 3 short, clear observations explaining what is happening
            and why it might be operationally relevant.
            """
        )

        chain = prompt | self.fast_llm | StrOutputParser()
        return chain.invoke({
            "rag_context": rag_context,
            "data_summary": data_summary
        })

    def generate_strategy(self, data_summary: str, _unused: str | None = None) -> str:
        rag_context = self._get_rag_context("forecast interpretation")

        prompt = ChatPromptTemplate.from_template(
            """
            You are a cautious analytical assistant interpreting forecasting outputs.

            HARD RULES (NON-NEGOTIABLE):
            - Do NOT suggest improving the model.
            - Do NOT suggest adding indicators or features.
            - Do NOT give financial or trading advice.
            - Do NOT assume predictive edge.
            - If signals are weak, explicitly state that.
            - Focus on interpreting signal characteristics, not restating missing information.
            - If the signal is weak, explain what types of decisions it is NOT suitable for.


            SOP CONTEXT:
            {rag_context}

            FORECAST CONTEXT:
            {data_summary}

            OUTPUT FORMAT (STRICT):
            Summary:
            Signal Strength:
            Risk Note:
            Appropriate Next Step:

            Keep the response factual, restrained, and professional.
            """
        )

        chain = prompt | self.smart_llm | StrOutputParser()
        return chain.invoke({
            "rag_context": rag_context,
            "data_summary": data_summary
        })