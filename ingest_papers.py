"""
Research Paper Ingestion Script for RAG System
Loads PDFs, chunks text, generates embeddings, and uploads to Pinecone
"""

import os
import time
from typing import List
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

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
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PDF_FOLDER = os.getenv("PDF_FOLDER", "./papers")

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100  # Pinecone batch upsert size


def initialize_pinecone():
    """Initialize Pinecone client and create index if needed"""
    logger.info("Initializing Pinecone...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Google embedding-001 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
        # Wait for index to be ready
        time.sleep(5)
        logger.info("Index created successfully")
    else:
        logger.info(f"Using existing index: {PINECONE_INDEX_NAME}")
    
    return pc.Index(PINECONE_INDEX_NAME)


def load_pdf_documents(pdf_directory: str) -> List:
    """Load all PDF files from directory"""
    logger.info(f"Loading PDFs from: {pdf_directory}")
    
    pdf_path = Path(pdf_directory)
    if not pdf_path.exists():
        raise ValueError(f"Directory not found: {pdf_directory}")
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_documents = []
    
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
            
            all_documents.extend(documents)
            logger.info(f"Loaded {len(documents)} pages from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {e}")
    
    logger.info(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents: List) -> List:
    """Split documents into smaller chunks"""
    logger.info("Chunking documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks


def create_embeddings_function():
    """Initialize Google Generative AI Embeddings"""
    logger.info("Initializing Google Embeddings API...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    return embeddings


def upsert_to_pinecone(index, chunks: List, embeddings):
    """Generate embeddings and upsert to Pinecone in batches"""
    logger.info("Generating embeddings and upserting to Pinecone...")
    
    total_chunks = len(chunks)
    
    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Upserting batches"):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        
        # Prepare batch data
        texts = [chunk.page_content for chunk in batch_chunks]
        metadatas = [chunk.metadata for chunk in batch_chunks]
        
        # Generate embeddings for batch
        try:
            batch_embeddings = embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i}: {e}")
            continue
        
        # Prepare vectors for Pinecone
        vectors = []
        for j, (text, embedding, metadata) in enumerate(zip(texts, batch_embeddings, metadatas)):
            vector_id = f"chunk_{i+j}_{metadata.get('source_file', 'unknown')}"
            
            # Add text to metadata for retrieval
            metadata["text"] = text
            metadata["chunk_index"] = i + j
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert to Pinecone
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            logger.error(f"Error upserting batch {i}: {e}")
            continue
    
    logger.info(f"Successfully upserted {total_chunks} chunks to Pinecone")


def main(pdf_directory: str):
    """Main ingestion pipeline"""
    logger.info("=" * 50)
    logger.info("Starting Research Paper Ingestion Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Initialize Pinecone
    index = initialize_pinecone()
    
    # Step 2: Load PDFs
    documents = load_pdf_documents(pdf_directory)
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    
    # Step 3: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 4: Initialize embeddings
    embeddings = create_embeddings_function()
    
    # Step 5: Upsert to Pinecone
    upsert_to_pinecone(index, chunks, embeddings)
    
    logger.info("=" * 50)
    logger.info("Ingestion completed successfully!")
    logger.info("=" * 50)
    
    # Print index stats
    stats = index.describe_index_stats()
    logger.info(f"Total vectors in index: {stats.total_vector_count}")


if __name__ == "__main__":
    main(PDF_FOLDER)