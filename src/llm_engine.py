import os
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

class LLMEngine:
    def __init__(self):
        print("[INFO] Initializing LLM Engine (via Groq Cloud)...")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")

        self.summarizer = ChatGroq(
            api_key=SecretStr(api_key),
            model="llama-3.1-8b-instant",
            temperature=0
        )
        
        self.strategist = ChatGroq(
            api_key=SecretStr(api_key),
            model="llama-3.3-70b-versatile", 
            temperature=0.2
        )
        
        self.vector_db_path = "models/chroma_db"
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None
        
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self): #sop to vector db
        if not os.path.exists("data/sops.txt"):
            print("CRITICAL ERROR: 'data/sops.txt' not found!")
            return

        if os.path.exists(self.vector_db_path):
            print("   -> Loading existing Vector DB...")
            self.db = Chroma(persist_directory=self.vector_db_path, embedding_function=self.embedding_function)
        else:
            print("   -> Creating new Vector DB from SOPs...")
            loader = TextLoader("data/sops.txt")
            docs = loader.load()
            self.db = Chroma.from_documents(documents=docs, embedding=self.embedding_function, persist_directory=self.vector_db_path)
            print("   -> Vector DB created successfully.")

    def summarize_reviews(self, reviews_list):
        if not reviews_list:
            return "No reviews available."
            
        print(f"   -> Summarizing {len(reviews_list)} reviews...")
        text_block = "\n".join(reviews_list[:30]) 
        
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the following reviews and extract key complaints.
            Be concise. Output a bulleted list.
            Reviews: {reviews}
            """
        )
        chain = prompt | self.summarizer | StrOutputParser()
        return chain.invoke({"reviews": text_block})

    def decide_action(self, anomaly_type, summary):
        print("   -> Consulting SOPs and Llama-3.3...")
        
        if self.db:
            query = f"{anomaly_type} {summary}"
            docs = self.db.similarity_search(query, k=1)
            retrieved_policy = docs[0].page_content if docs else "No specific policy found."
        else:
            retrieved_policy = "No Knowledge Base loaded."
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are an Operations Manager.
            CONTEXT:
            - Issue: {anomaly_type}
            - Feedback: {summary}
            
            POLICY:
            {policy}
            
            TASK:
            Determine the action based on policy. Draft a short ticket.
            """
        )
        chain = prompt | self.strategist | StrOutputParser()
        return chain.invoke({"anomaly_type": anomaly_type, "summary": summary, "policy": retrieved_policy})