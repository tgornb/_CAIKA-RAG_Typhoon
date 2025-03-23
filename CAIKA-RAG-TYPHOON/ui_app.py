# streamlit_app.py for CAIKA-RAG-Typhoon
import streamlit as st
import os
import logging
from pathlib import Path
from src.utils.document_loader import load_documents, split_documents
from src.embeddings.embeddings import EmbeddingManager
from src.llm.llm_manager import LLMManager
from src.retrieval.rag_pipeline import RAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("caika_web.log")]
)
logger = logging.getLogger("caika_web")

# Set page configuration
st.set_page_config(
    page_title="CAIKA-RAG-Typhoon",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 24px;
        color: #424242;
        margin-bottom: 30px;
    }
    .stAlert {
        background-color: #e8f4f8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []

def initialize_system(model_path="models/llama3.2-typhoon2-1b.gguf"):
    """Initialize the CAIKA-RAG-Typhoon system."""
    try:
        with st.spinner("Initializing CAIKA-RAG-Typhoon..."):
            # Setup knowledge base
            documents_dir = "data/documents"
            vector_store_path = "data/vector_store"
            
            # Ensure directories exist
            Path(documents_dir).mkdir(parents=True, exist_ok=True)
            Path(vector_store_path).mkdir(parents=True, exist_ok=True)
            
            embedding_manager = EmbeddingManager()
            
            # Check if vector store exists
            if os.path.exists(vector_store_path) and any(os.scandir(vector_store_path)):
                logger.info("Loading existing vector store...")
                vector_store = embedding_manager.load_vector_store(vector_store_path)
                st.session_state.vector_store = vector_store
            else:
                if any(os.scandir(documents_dir)):
                    # Process existing documents
                    logger.info("No existing vector store. Creating from available documents...")
                    documents = load_documents(documents_dir)
                    chunks = split_documents(documents)
                    vector_store = embedding_manager.create_vector_store(chunks, vector_store_path)
                    st.session_state.vector_store = vector_store
                    logger.info(f"Created vector store from {len(documents)} documents.")
                else:
                    st.warning("No documents found in data/documents directory. Please upload documents.")
                    return False
            
            # Setup LLM
            logger.info(f"Initializing LLM with model: {model_path}")
            llm_manager = LLMManager(model_path=model_path)
            llm = llm_manager.get_llm()
            
            # Setup RAG pipeline
            rag = RAGPipeline(st.session_state.vector_store, llm)
            st.session_state.qa_chain = rag.create_qa_chain()
            
            st.session_state.initialized = True
            logger.info("CAIKA-RAG-Typhoon initialized successfully.")
            return True
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        st.error(f"Error initializing system: {str(e)}")
        return False

# Sidebar for system controls
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>CAIKA-RAG-Typhoon</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title' style='text-align: center;'>Corporate AI Knowledge Assistant</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Model selection
    model_path = st.text_input(
        "Model Path", 
        value="models/llama3.2-typhoon2-1b.gguf",
        help="Path to your LLM model file"
    )
    
    # LLM parameters
    with st.expander("LLM Parameters"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05)
        max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 64)
    
    if st.button("Initialize CAIKA"):
        success = initialize_system(model_path)
        if success:
            st.success("CAIKA-RAG-Typhoon initialized successfully!")
    
    st.divider()
    
    # Document uploader section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload corporate documents (PDFs)", 
                                     type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Uploaded Documents"):
        with st.spinner("Processing documents..."):
            try:
                # Create documents directory if it doesn't exist
                os.makedirs("data/documents", exist_ok=True)
                
                # Save uploaded files
                for file in uploaded_files:
                    file_path = os.path.join("data/documents", file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                logger.info(f"Saved {len(uploaded_files)} uploaded files.")
                
                # Load and process documents
                documents = load_documents("data/documents")
                chunks = split_documents(documents)
                
                logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
                
                # Create/update vector store
                embedding_manager = EmbeddingManager()
                vector_store = embedding_manager.create_vector_store(chunks)
                
                # Update session state
                st.session_state.vector_store = vector_store
                
                # Setup LLM if not already done
                if not st.session_state.initialized:
                    llm_manager = LLMManager(model_path=model_path)
                    llm = llm_manager.get_llm()
                    
                    # Setup RAG pipeline
                    rag = RAGPipeline(vector_store, llm)
                    st.session_state.qa_chain = rag.create_qa_chain()
                    st.session_state.initialized = True
                
                st.success(f"Processed {len(documents)} documents into {len(chunks)} chunks.")
                logger.info("Document processing completed successfully.")
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                st.error(f"Error processing documents: {str(e)}")

# Main content area
st.markdown("<h1 class='main-title'>CAIKA-RAG-Typhoon</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Your Corporate AI Knowledge Assistant</div>", unsafe_allow_html=True)

# Check if system is initialized
if not st.session_state.initialized:
    st.info("CAIKA-RAG-Typhoon is not initialized. Please click 'Initialize CAIKA' in the sidebar to start.")
    
    # Show setup instructions
    with st.expander("Setup Instructions", expanded=True):
        st.markdown("""
        ### Getting Started with CAIKA-RAG-Typhoon:
        
        1. **Upload Documents**: Use the sidebar to upload corporate documents (PDFs).
        2. **Process Documents**: Click "Process Uploaded Documents" to index your knowledge base.
        3. **Initialize CAIKA**: Click "Initialize CAIKA" to start the assistant.
        4. **Ask Questions**: Once initialized, you can ask questions about your corporate knowledge.
        
        CAIKA-RAG-Typhoon uses the llama-cpp-python library optimized for Apple Silicon, providing efficient performance on your MacBook Air M1.
        """)
else:
    # Create two columns for the chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_input("Ask a question about your corporate knowledge:")
        
        if query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("CAIKA is thinking..."):
                try:
                    logger.info(f"Processing query: {query}")
                    result = st.session_state.qa_chain({"query": query})
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": result["result"]})
                    
                    # Store source documents for this query
                    st.session_state.last_sources = result["source_documents"][:3]
                    
                    logger.info("Query processed successfully.")
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    logger.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I encountered an error: {error_msg}"})
            
        # Display chat history
        st.subheader("Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**CAIKA:** {message['content']}")
            st.divider()
    
    with col2:
        # Show sources for the last query if available
        if st.session_state.chat_history and hasattr(st.session_state, 'last_sources'):
            st.subheader("Sources")
            for i, doc in enumerate(st.session_state.last_sources):
                source = doc.metadata.get('source', 'Unknown source')
                st.markdown(f"**Source {i+1}:** {os.path.basename(source)}")
                
                # Show a snippet of the content
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                with st.expander("Content preview"):
                    st.write(content_preview)