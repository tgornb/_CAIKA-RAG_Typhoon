# src/embeddings/embeddings.py
import os
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

logger = logging.getLogger("caika.embeddings")

class EmbeddingManager:
    """Manager for handling embeddings and vector stores."""
    
    def __init__(self, model_name="scb10x/llama3.2-typhoon2-t1-3b-research-preview", device="cpu"):
        """
        Initialize the Embedding Manager.
        
        Args:
            model_name (str): Name of the HuggingFace embedding model.
            device (str): Device to use for embeddings (mps for M1 Mac).
        """
        logger.info(f"Initializing embeddings with model {model_name} on device {device}")
        
        # Explicitly check that model_name is a valid embedding model
        if "gguf" in model_name.lower():
            raise ValueError("GGUF files cannot be used as embedding models. Please use a Hugging Face embedding model.")

        # For M1 Mac, use the MPS (Metal Performance Shaders) device when available
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
        
    def create_vector_store(self, documents, persist_directory="data/vector_store"):
        """
        Create a vector store from documents.
        
        Args:
            documents (list): List of Document objects.
            persist_directory (str): Directory to save the vector store.
            
        Returns:
            FAISS: The FAISS vector store.
        """
        try:
            logger.info(f"Creating vector store from {len(documents)} documents")
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            # Save vector store
            vectorstore.save_local(persist_directory)
            logger.info(f"Vector store saved to {persist_directory}")
            
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self, persist_directory="data/vector_store"):
        """
        Load a persisted vector store.
        
        Args:
            persist_directory (str): Directory where the vector store is saved.
            
        Returns:
            FAISS: The loaded FAISS vector store.
        """
        try:
            if os.path.exists(persist_directory):
                logger.info(f"Loading vector store from {persist_directory}")
                # Add allow_dangerous_deserialization=True parameter
                return FAISS.load_local(
                persist_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Add this parameter
            )
            else:
                error_msg = f"No vector store found at {persist_directory}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
