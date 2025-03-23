# src/retrieval/rag_pipeline.py
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger("caika.rag_pipeline")

class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) Pipeline."""
    
    def __init__(self, vector_store, llm, search_kwargs=None):
        """
        Initialize the RAG Pipeline.
        
        Args:
            vector_store: The vector store for retrieval.
            llm: The language model for generation.
            search_kwargs (dict, optional): Arguments for the retriever search.
        """
        self.vector_store = vector_store
        self.llm = llm
        
        # Default search parameters
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        # Initialize retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        logger.info("RAG Pipeline initialized successfully")
        
    def create_qa_chain(self):
        """
        Create a question-answering chain with RAG.
        
        Returns:
            RetrievalQA: The initialized QA chain.
        """
        try:
            # Define a template with clear instructions
            template = """
            You are CAIKA, a friendly and helpful Corporate AI Knowledge Assistant. Your goal is to provide accurate, 
            helpful answers based on the provided context from corporate documents.
            
            Instructions:
            1. Use ONLY the information in the provided context to answer the question.
            2. If the context doesn't contain the answer, respond with a friendly message explaining that you don't have 
               enough information and suggest what kind of document might contain this information.
            3. Format your responses using markdown to improve readability - use headers, bullet points, and emphasis where appropriate.
            4. When citing information, mention which document it comes from (e.g., "According to [document name],...").
            5. Keep your tone conversational and helpful - like a knowledgeable colleague.
            6. At the end of detailed responses, include a brief 1-2 sentence summary of the key points.
            7. For complex answers, structure information in a logical flow from general to specific details.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            # Create the prompt template
            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create the chain
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Simple method that stuffs all documents into the prompt
                retriever=self.retriever,
                return_source_documents=True,  # Return the source documents for reference
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            logger.info("QA chain created successfully")
            return chain
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
            
    def update_retriever_parameters(self, k=None, fetch_k=None, score_threshold=None):
        """
        Update retriever parameters.
        
        Args:
            k (int, optional): Number of documents to retrieve.
            fetch_k (int, optional): Number of documents to fetch before filtering.
            score_threshold (float, optional): Minimum similarity score threshold.
            
        Returns:
            Retriever: The updated retriever.
        """
        try:
            search_kwargs = {}
            
            if k is not None:
                search_kwargs["k"] = k
                
            if fetch_k is not None:
                search_kwargs["fetch_k"] = fetch_k
                
            if score_threshold is not None:
                search_kwargs["score_threshold"] = score_threshold
                
            if search_kwargs:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs=search_kwargs
                )
                logger.info(f"Updated retriever parameters: {search_kwargs}")
                
            return self.retriever
        except Exception as e:
            logger.error(f"Error updating retriever parameters: {str(e)}")
            raise
