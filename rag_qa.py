"""
Research Paper QA RAG System (Modernized)
Retrieval-Augmented Generation for answering questions about research papers
Using LangChain Expression Language (LCEL) and latest patterns
"""

import os
import logging
from dotenv import load_dotenv
from typing import List, Dict
from operator import itemgetter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()



# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "telecoms-papers")

# Retrieval parameters
TOP_K_RESULTS = 5
RELEVANCE_THRESHOLD = 0.78


def initialize_embeddings():
    """Initialize Google Generative AI Embeddings"""
    logger.info("Initializing embeddings...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    return embeddings


def initialize_llm():
    """Initialize Gemini 2.5 Flash LLM"""
    logger.info("Initializing Gemini 2.5 Flash LLM...")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Latest Gemini Flash model
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    return llm


def initialize_vector_store(embeddings):
    """Connect to Pinecone vector store"""
    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Create LangChain vector store wrapper
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"  # The metadata field containing the text
    )
    
    logger.info("Vector store initialized successfully")
    return vectorstore


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        content = doc.page_content
        formatted.append(
            f"[Source {i}: {source}, Page {page}]\n{content}\n"
        )
    return "\n".join(formatted)


def create_rag_prompt():
    """Create modern chat prompt template for RAG"""
    template = """Role & Objective: You are a knowledgeable and reliable research assistant. 
    Use the provided retrieved context to answer the user’s question accurately, clearly, and concisely.
    Also provided is the History of conversation thus far, use only if necessary
    

Instructions:

Always prioritize the retrieved context over your own knowledge.
If the context does not contain enough information, say so and avoid guessing.
Do not fabricate facts or add unsupported details.
Keep responses well-structured, easy to read, and relevant to the user’s query.
Maintain a friendly, professional, and helpful tone.

Context Boundaries:

Only use information from the retrieved documents unless it’s general, widely accepted knowledge.
Only use use context that is relevant to the question asked.

Context: {context}

history: {history}

Question: {question}

Output Format:

Direct answer first.
Optional short explanation or reasoning.
Use bullet points or headings for clarity when needed.
"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_rag_chain(vectorstore, llm):
    """Create modern RAG chain using LCEL (LangChain Expression Language)"""
    logger.info("Creating RAG chain with LCEL...")
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",#"similarity",
        search_kwargs={"k": TOP_K_RESULTS, "score_threshold": RELEVANCE_THRESHOLD}
    )
    
    # Create prompt
    prompt = create_rag_prompt()
    
    # Build RAG chain using LCEL
    rag_chain = (
        RunnableParallel(
            context=itemgetter("question") | retriever | format_docs,
            question=itemgetter("question"),
            history=itemgetter("history")
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain created successfully")
    return rag_chain, retriever


def format_sources(source_documents: List[Document]) -> str:
    """Format source documents for display"""
    sources = []
    seen = set()  # Avoid duplicate sources
    
    for i, doc in enumerate(source_documents, 1):
        source_file = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        source_key = f"{source_file}_{page}"
        
        if source_key not in seen:
            sources.append(f"[{len(sources) + 1}] {source_file} (Page {page})")
            seen.add(source_key)
    
    return "  \n".join(sources) if sources else ""

message_map = {
    "assistant": AIMessage,
    "user": HumanMessage
}
def get_response(rag_chain, retriever, question: str, messages: str) -> str:
    """Ask a question and get answer with sources using modern LCEL chain"""
    logger.info(f"Processing question: {question}")
    
    try:
        history = []
        
        for m in messages:
            history.append(message_map[m["role"]](m["content"]))

        logger.info(f"History: {history}")
        # Retrieve source documents separately for display
        source_docs = retriever.invoke(question)
        
        # Run the RAG chain
        answer = rag_chain.invoke({"question": question, "history":history})
        
        logger.info("Answer generated successfully")
        
        return answer+"\n\n"+ format_sources(source_docs)
    
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"Error: {str(e)}"+"\n\n"+ "Error retrieving sources"
