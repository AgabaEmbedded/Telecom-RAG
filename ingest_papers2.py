"""
Research Paper Ingestion Script (HuggingFace Version)
Loads PDFs, chunks text, generates embeddings using sentence-transformers, uploads to Pinecone
No LangChain dependencies
"""

import os
import time
from typing import List, Dict
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-papers")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions, fast and efficient

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100  # Pinecone batch upsert size


def initialize_embedding_model():
    """Initialize sentence transformer model"""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {dimension}")
    return model, dimension


def initialize_pinecone(dimension: int):
    """Initialize Pinecone client and create index if needed"""
    logger.info("Initializing Pinecone...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
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


def load_pdf_documents(pdf_directory: str) -> List[Dict]:
    """Load all PDF files from directory and extract text"""
    logger.info(f"Loading PDFs from: {pdf_directory}")
    
    pdf_path = Path(pdf_directory)
    if not pdf_path.exists():
        raise ValueError(f"Directory not found: {pdf_directory}")
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_documents = []
    
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            reader = PdfReader(str(pdf_file))
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                if text.strip():  # Only add if text is not empty
                    document = {
                        'text': text,
                        'metadata': {
                            'source_file': pdf_file.name,
                            'page': page_num + 1,  # 1-indexed
                            'total_pages': len(reader.pages)
                        }
                    }
                    all_documents.append(document)
            
            logger.info(f"Loaded {len(reader.pages)} pages from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {e}")
    
    logger.info(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep periods, commas, etc.
    text = re.sub(r'[^\w\s.,!?;:()\-\[\]{}\'\"]+', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    text = clean_text(text)
    
    if len(text) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            last_period = text.rfind('.', start, end)
            last_question = text.rfind('?', start, end)
            last_exclamation = text.rfind('!', start, end)
            
            break_point = max(last_period, last_question, last_exclamation)
            
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Split all documents into smaller chunks"""
    logger.info("Chunking documents...")
    
    all_chunks = []
    
    for doc in tqdm(documents, desc="Chunking"):
        text = doc['text']
        metadata = doc['metadata']
        
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'text': chunk,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            all_chunks.append(chunk_doc)
    
    logger.info(f"Created {len(all_chunks)} chunks")
    return all_chunks


def upsert_to_pinecone(index, chunks: List[Dict], embedding_model):
    """Generate embeddings and upsert to Pinecone in batches"""
    logger.info("Generating embeddings and upserting to Pinecone...")
    
    total_chunks = len(chunks)
    
    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Upserting batches"):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        
        # Extract texts
        texts = [chunk['text'] for chunk in batch_chunks]
        
        # Generate embeddings for batch
        try:
            batch_embeddings = embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_tensor=False
            )
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i}: {e}")
            continue
        
        # Prepare vectors for Pinecone
        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            metadata = chunk['metadata']
            text = chunk['text']
            
            # Create unique ID
            vector_id = f"chunk_{i+j}_{metadata['source_file']}_{metadata['page']}"
            
            # Add text to metadata
            metadata['text'] = text
            
            vectors.append({
                "id": vector_id,
                "values": embedding.tolist(),
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
    logger.info("Using HuggingFace Sentence Transformers")
    logger.info("=" * 50)
    
    # Step 1: Initialize embedding model
    embedding_model, dimension = initialize_embedding_model()
    
    # Step 2: Initialize Pinecone
    index = initialize_pinecone(dimension)
    
    # Step 3: Load PDFs
    documents = load_pdf_documents(pdf_directory)
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    
    # Step 4: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 5: Upsert to Pinecone
    upsert_to_pinecone(index, chunks, embedding_model)
    
    logger.info("=" * 50)
    logger.info("Ingestion completed successfully!")
    logger.info("=" * 50)
    
    # Print index stats
    stats = index.describe_index_stats()
    logger.info(f"Total vectors in index: {stats.total_vector_count}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingest_papers_hf.py <pdf_directory>")
        print("Example: python ingest_papers_hf.py ./papers")
        sys.exit(1)
    
    pdf_dir = sys.argv[1]
    main(pdf_dir)