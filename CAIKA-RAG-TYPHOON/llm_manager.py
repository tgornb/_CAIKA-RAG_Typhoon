# src/llm/llm_manager.py - Modified for Hugging Face models
import logging, os
from langchain_community.llms import LlamaCpp  # Make sure to use this import


logger = logging.getLogger("caika.llm_manager")

class LLMManager:
    """Manager for LLM models."""
    
    def __init__(self, 
                model_path="models/scb10x_llama3.2-typhoon2-t1-3b-research-preview-gguf_llama3.2-typhoon2-t1-3b-q4_k_m.gguf",  # Path or HF model ID
                temperature=0.7,
                max_tokens=2048,
                top_p=0.95):
        # Expand user path (for ~)
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Check if file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n_ctx=2048,  # Reduced context window to save memory
                n_gpu_layers=-1,  # Use all available GPU layers
                verbose=False,
                n_threads=4,  # Adjust based on your M1 CPU cores
                f16_kv=True,  # Use half-precision for key/value cache
            )
            logger.info("LlamaCpp model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise


        """
        Initialize the LLM Manager for Hugging Face models.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
    
    def get_llm(self):
        """Get the initialized LLM model."""
        return self.llm