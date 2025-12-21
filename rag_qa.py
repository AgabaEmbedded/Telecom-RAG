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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-papers")

# Retrieval parameters
TOP_K_RESULTS = 5


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
        model="gemini-2.0-flash-exp",  # Latest Gemini Flash model
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
    template = """You are an expert research assistant analyzing academic papers. 
Use the following pieces of context from research papers to answer the question.

If you don't know the answer based on the provided context, just say that you don't know. 
Don't try to make up an answer.

Always cite the source document(s) and page numbers in your answer when possible.

Context from research papers:
{context}

Question: {question}

Provide a detailed, well-structured answer based on the context above:"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_rag_chain(vectorstore, llm):
    """Create modern RAG chain using LCEL (LangChain Expression Language)"""
    logger.info("Creating RAG chain with LCEL...")
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    # Create prompt
    prompt = create_rag_prompt()
    
    # Build RAG chain using LCEL
    rag_chain = (
        RunnableParallel(
            context=itemgetter("question") | retriever | format_docs,
            question=itemgetter("question")
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
    
    return "\n\n".join(sources) if sources else "No sources available"


def ask_question(rag_chain, retriever, question: str) -> Dict:
    """Ask a question and get answer with sources using modern LCEL chain"""
    logger.info(f"Processing question: {question}")
    
    try:
        # Retrieve source documents separately for display
        source_docs = retriever.invoke(question)
        
        # Run the RAG chain
        answer = rag_chain.invoke({"question": question})
        
        logger.info("Answer generated successfully")
        
        return {
            "answer": answer,
            "sources": source_docs,
            "formatted_sources": format_sources(source_docs)
        }
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "formatted_sources": "Error retrieving sources"
        }


def interactive_qa_loop(rag_chain, retriever):
    """Interactive question-answering loop"""
    print("\n" + "=" * 60)
    print("Research Paper QA System - Interactive Mode")
    print("=" * 60)
    print("Ask questions about your research papers!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("\n🔍 Your Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Research Paper QA System!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\n⏳ Searching and generating answer...\n")
        
        result = ask_question(rag_chain, retriever, question)
        
        print("-" * 60)
        print("📝 ANSWER:")
        print("-" * 60)
        print(result["answer"])
        print("\n" + "-" * 60)
        print("📚 SOURCES:")
        print("-" * 60)
        print(result["formatted_sources"])
        print("=" * 60)


def main():
    """Main RAG QA pipeline"""
    logger.info("=" * 50)
    logger.info("Starting Research Paper QA RAG System")
    logger.info("=" * 50)
    
    # Step 1: Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Step 2: Initialize LLM
    llm = initialize_llm()
    
    # Step 3: Initialize vector store
    vectorstore = initialize_vector_store(embeddings)
    
    # Step 4: Create RAG chain (returns chain and retriever)
    rag_chain, retriever = create_rag_chain(vectorstore, llm)
    
    logger.info("System ready!")
    
    # Step 5: Start interactive QA
    interactive_qa_loop(rag_chain, retriever)

def get_response(rag_chain, retriever, question: str) -> str:
    """Ask a question and get answer with sources using modern LCEL chain"""
    logger.info(f"Processing question: {question}")
    
    try:
        # Retrieve source documents separately for display
        source_docs = retriever.invoke(question)
        
        # Run the RAG chain
        answer = rag_chain.invoke({"question": question})
        
        logger.info("Answer generated successfully")
        
        return answer+"\n\n"+ format_sources(source_docs)
    
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"Error: {str(e)}"+"\n\n"+ "Error retrieving sources"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
    else:
        # Interactive mode
        main()