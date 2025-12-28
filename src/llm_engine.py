import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

class LLMEngine:
    def __init__(self):
        print("[INFO] Initializing LLM Engine...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env!")

        # TIER 1: FAST (Llama-3.1-8b-instant)
        # Note: Groq serves the base model. To use a custom LoRA, 
        # you typically need a local server (like vLLM) or a platform that supports adapters.
        # For this "Show Off", we will simulate the "Fine-Tuned" style via sophisticated prompting.
        self.fast_llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.3)
        
        # TIER 2: SMART (Llama-3.3-70b-versatile)
        self.smart_llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.3)
        
        # RAG Setup
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None

    def update_knowledge_base(self, text_content):
        """Dynamically builds the vector DB from user input."""
        if not text_content.strip():
            self.vector_db = None
            return "[INFO] Knowledge Base cleared."

        print(" -> Updating Knowledge Base...")
        doc = Document(page_content=text_content, metadata={"source": "user_input"})
        self.vector_db = Chroma.from_documents(documents=[doc], embedding=self.embedding_function)
        return "[SUCCESS] RAG System Updated."

    def _get_rag_context(self, query):
        if self.vector_db:
            docs = self.vector_db.similarity_search(query, k=1)
            if docs:
                return docs[0].page_content
        return "No specific rules."

    def analyze_visuals(self, data_summary):
        """
        TIER 1: "Fine-Tuned Style" Analysis.
        Instead of repeating numbers, it focuses on business implications.
        """
        rag_context = self._get_rag_context("visual trends summary")
        
        # WE 'SHOW OFF' BY GIVING IT A PERSONA THAT MIMICS A FINE-TUNED ANALYST
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Senior Data Journalist using a specific style called 'The Insightful LoRA'.
            
            STYLE GUIDELINES:
            1. DO NOT repeat the exact numbers from the data summary unless necessary for contrast.
            2. Focus on the 'Why' and 'So What'.
            3. Use professional, punchy language. No "The data indicates..."
            4. If a User Rule (SOP) is present, mention how the data respects or violates it.
            
            USER RULES: {rag_context}
            
            DATA SUMMARY:
            {data_summary}
            
            OUTPUT:
            Provide 3 sharp, distinct insights in the 'Insightful LoRA' style.
            """
        )
        
        chain = prompt | self.fast_llm | StrOutputParser()
        return chain.invoke({"rag_context": rag_context, "data_summary": data_summary})

    def generate_strategy(self, data_summary, user_query=""):
        """
        TIER 2: Strategic Advice (70b)
        """
        rag_context = self._get_rag_context(user_query + " forecast strategy")
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Strategy Consultant.
            CONTEXT: {rag_context}
            DATA: {data_summary}
            
            TASK: Provide 3 strategic, actionable steps.
            """
        )
        chain = prompt | self.smart_llm | StrOutputParser()
        return chain.invoke({"rag_context": rag_context, "data_summary": data_summary})