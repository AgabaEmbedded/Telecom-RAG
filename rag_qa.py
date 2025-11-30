"""
Research Paper QA RAG System
Retrieval-Augmented Generation for answering questions about research papers
"""

import os
import logging
from dotenv import load_dotenv
from typing import List, Dict

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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


def create_qa_prompt():
    """Create custom prompt template for QA"""
    template = """You are an expert research assistant analyzing academic papers. 
Use the following pieces of context from research papers to answer the question at the end.

If you don't know the answer based on the provided context, just say that you don't know. 
Don't try to make up an answer.

Always cite the source document(s) in your answer when possible.

Context:
{context}

Question: {question}

Detailed Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_rag_chain(vectorstore, llm):
    """Create the RAG retrieval chain"""
    logger.info("Creating RAG chain...")
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    # Create custom prompt
    prompt = create_qa_prompt()
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' passes all docs to LLM at once
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    logger.info("RAG chain created successfully")
    return qa_chain


def format_sources(source_documents: List) -> str:
    """Format source documents for display"""
    sources = []
    for i, doc in enumerate(source_documents, 1):
        source_file = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        sources.append(f"[{i}] {source_file} (Page {page})")
    
    return "\n".join(sources)


def ask_question(qa_chain, question: str) -> Dict:
    """Ask a question and get answer with sources"""
    logger.info(f"Processing question: {question}")
    
    try:
        # Run the chain
        result = qa_chain.invoke({"query": question})
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
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
            "formatted_sources": ""
        }


def interactive_qa_loop(qa_chain):
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
        
        result = ask_question(qa_chain, question)
        
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
    
    # Step 4: Create RAG chain
    qa_chain = create_rag_chain(vectorstore, llm)
    
    logger.info("System ready!")
    
    # Step 5: Start interactive QA
    interactive_qa_loop(qa_chain)


def single_question_mode(question: str):
    """Process a single question (for scripting)"""
    logger.info("Running in single question mode")
    
    embeddings = initialize_embeddings()
    llm = initialize_llm()
    vectorstore = initialize_vector_store(embeddings)
    qa_chain = create_rag_chain(vectorstore, llm)
    
    result = ask_question(qa_chain, question)
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    print(result["formatted_sources"])
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
    else:
        # Interactive mode
        main()