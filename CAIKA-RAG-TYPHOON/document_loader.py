# src/utils/document_loader.py
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger("caika.document_loader")

def load_documents(directory_path):
    """
    Load documents from a directory.
    
    Args:
        directory_path (str): Path to the directory containing documents.
        
    Returns:
        list: List of Document objects.
    """
    try:
        loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {directory_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from {directory_path}: {str(e)}")
        raise

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into manageable chunks for embedding.
    
    Args:
        documents (list): List of Document objects.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks to maintain context.
        
    Returns:
        list: List of Document chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise
