#!/usr/bin/env python3
# app.py for CAIKA-RAG-Typhoon

import os, sys
import argparse
import logging
from pathlib import Path
# Get the absolute path to your project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the project root to Python's sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Direct imports from your modules
from document_loader import load_documents, split_documents
from embeddings_manager import EmbeddingManager
from llm_manager import LLMManager
from rag_pipeline import RAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("caika.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("caika")

def setup_knowledge_base(documents_dir, force_reload=False):
    """Set up the knowledge base."""
    vector_store_path = "data/vector_store"
    embedding_manager = EmbeddingManager()
    
    # Check if we need to rebuild the vector store
    if force_reload or not os.path.exists(vector_store_path) or not any(os.scandir(vector_store_path)):
        logger.info("Loading documents from %s...", documents_dir)
        documents = load_documents(documents_dir)
        logger.info(f"Loaded {len(documents)} documents.")
        
        logger.info("Splitting documents into chunks...")
        chunks = split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")
        
        logger.info("Creating vector store...")
        vector_store = embedding_manager.create_vector_store(chunks, vector_store_path)
        logger.info("Vector store created successfully.")
    else:
        logger.info("Loading existing vector store...")
        vector_store = embedding_manager.load_vector_store(vector_store_path)
        logger.info("Vector store loaded successfully.")
    
    return vector_store

def main():
    parser = argparse.ArgumentParser(description="CAIKA-RAG-Typhoon - Corporate AI Knowledge Assistant")
    parser.add_argument("--docs", type=str, default="data/documents", 
                        help="Path to documents directory")
    parser.add_argument("--rebuild", action="store_true", 
                        help="Force rebuild of the vector store")
    parser.add_argument("--model", type=str, default="models/llama3.2-typhoon2-1b.gguf",
                        help="Path to the model file (.gguf format)")
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    Path("data/vector_store").mkdir(parents=True, exist_ok=True)
    
    # Print welcome message
    print("""
    ╔═══════════════════════════════════════════╗
    ║           CAIKA-RAG-Typhoon               ║
    ║     Corporate AI Knowledge Assistant      ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # Setup knowledge base
    vector_store = setup_knowledge_base(args.docs, args.rebuild)
    
    # Setup LLM
    logger.info(f"Initializing LLM with model: {args.model}")
    llm_manager = LLMManager(model_path=args.model)
    llm = llm_manager.get_llm()
    
    # Setup RAG pipeline
    rag = RAGPipeline(vector_store, llm)
    qa_chain = rag.create_qa_chain()
    
    # Interactive query loop
    print("\nCAIKA is ready! Type 'exit' to quit.")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        try:
            logger.info(f"Processing query: {query}")
            result = qa_chain({"query": query})
            
            print("\nAnswer:", result["result"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"][:2]):
                source = doc.metadata.get('source', 'Unknown source')
                print(f"- {source}")
                
            logger.info(f"Query answered successfully")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()