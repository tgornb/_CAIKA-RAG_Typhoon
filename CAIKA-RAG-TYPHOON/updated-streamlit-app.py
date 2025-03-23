# streamlit_app.py for CAIKA-RAG-Typhoon
# At the top of app.py and streamlit_app.py
import os
from pathlib import Path
import sys
import base64
from PIL import Image
import io

def add_logo():
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    logo_path = "assets/caika_logo.png"
    
    # Check if the logo exists
    if os.path.exists(logo_path):
        # Use Streamlit's column layout for centering
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo_path, width=250)
    else:
        st.warning("Please save the CAIKA logo to assets/caika_logo.png")

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

import streamlit as st
import logging

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

# Custom CSS with improved chat styling and bottom input
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
    .chat-container {
        border-radius: 10px;
        margin-bottom: 12px;
        padding: 2px;
    }
    .user-message {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        position: relative;
        border-left: 5px solid #7E57C2;
    }
    .assistant-message {
        background-color: #e8f4fa;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        position: relative;
        border-left: 5px solid #1E88E5;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .source-tag {
        display: inline-block;
        background-color: #E0E0E0;
        border-radius: 5px;
        padding: 2px 8px;
        margin-right: 5px;
        font-size: 0.8em;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF8F00;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    .input-container {
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []

def initialize_system(model_path="models/scb10x_llama3.2-typhoon2-t1-3b-research-preview-gguf_llama3.2-typhoon2-t1-3b-q4_k_m.gguf"):
    """Initialize the CAIKA-RAG-Typhoon system."""
    try:
        with st.spinner("Initializing CAIKA-RAG-Typhoon..."):
            # Setup knowledge base
            documents_dir = "data/documents"
            vector_store_path = "data/vector_store"
            
            # Ensure directories exist
            Path(documents_dir).mkdir(parents=True, exist_ok=True)
            Path(vector_store_path).mkdir(parents=True, exist_ok=True)
            
             # IMPORTANT: Initialize embedding manager with a specific embedding model, NOT your LLM model path
            embedding_manager = EmbeddingManager(model_name="scb10x/llama3.2-typhoon2-t1-3b-research-preview")
            
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
    add_logo()
    st.markdown("<div class='sub-title' style='text-align: center;'>Corporate AI Knowledge Assistant Optimized for Apple M1 (CPU based)</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Model selection
    model_path = st.text_input(
        "Model Path", 
        value="models/scb10x_llama3.2-typhoon2-t1-3b-research-preview-gguf_llama3.2-typhoon2-t1-3b-q4_k_m.gguf",
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
                vector_store = embedding_manager.create_vector_store(chunks, "data/vector_store")
                
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
st.markdown("<div class='sub-title'>Your Friendly Corporate AI Knowledge Assistant by <b>Thapagorn Hanrak</b></div>", unsafe_allow_html=True)

# Check if system is initialized
if not st.session_state.initialized:
    st.info("CAIKA-RAG-Typhoon is not initialized. Please click 'Initialize CAIKA' in the sidebar to start.")
    
    # Show setup instructions
    with st.expander("Setup Instructions", expanded=True):
        st.markdown("""
        ### Getting Started with CAIKA-RAG-Typhoon:
        
        1. **Initialize CAIKA**: Click "Initialize CAIKA" to start the assistant.
        2. **Upload Documents**: Use the sidebar to upload corporate documents (PDFs).
        3. **Process Documents**: Click "Process Uploaded Documents" to index your knowledge base.
        4. **Ask Questions**: Once initialized, you can ask questions about your corporate knowledge.
        
        CAIKA-RAG-Typhoon uses the llama-cpp-python library optimized for Apple Silicon and local model **scb10x_llama3.2-typhoon2-t1-3b**, providing efficient performance on your MacBook Air M1. 
        Thanks **SCB10X** for your model. Please contact me via **thapagorn.h@outlook.com** if you have any questions or need help.
        """)
else:
    # Create two columns for the chat interface with improved ratio
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display conversation history at the top
        st.subheader("Conversation")
        
        # Container for scrollable chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-container">
                        <div class="user-message">
                            <div class="message-header">You</div>
                            {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # For assistant responses with the new format
                    if isinstance(message["content"], dict):
                        content = message["content"]["content"]
                        sources = message["content"].get("sources", [])
                        confidence = message["content"].get("confidence", "medium")
                        
                        # Format sources as tags
                        source_tags = ""
                        if sources:
                            source_tags = "Sources: " + " ".join([f'<span class="source-tag">{source}</span>' for source in sources])
                        
                        # Format confidence indicator
                        confidence_indicator = f'<span class="confidence-{confidence}">Confidence: {confidence.title()}</span>'
                        
                        st.markdown(f"""
                        <div class="chat-container">
                            <div class="assistant-message">
                                <div class="message-header">CAIKA</div>
                                {content}
                                <hr>
                                <div>{source_tags} {confidence_indicator}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # For older format messages (backward compatibility)
                        st.markdown(f"""
                        <div class="chat-container">
                            <div class="assistant-message">
                                <div class="message-header">CAIKA</div>
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Add a visual spacer
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # NOW - Add input box at the bottom, in a container with special styling
        with st.container():
            st.markdown('<div class="input-container"></div>', unsafe_allow_html=True)
            
            # Example question buttons for quick access
            cols = st.columns(4)
            if cols[0].button("What is CAIKA?"):
                query = "What is CAIKA-RAG-Typhoon and what does it do?"
            elif cols[1].button("How to use?"):
                query = "How do I get better results from the assistant?"
            elif cols[2].button("Upload docs?"):
                query = "How do I upload and process new documents?"
            elif cols[3].button("System specs?"):
                query = "What are the technical specifications of the model?"
            else:
                # Regular input if no quick button was pressed
                query = st.text_input("Ask CAIKA a question...", key="query_input", placeholder="Type your question here and press Enter")
        
        # Process the query
        if query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("CAIKA is กำลังคิด..."):
                try:
                    logger.info(f"Processing query: {query}")
                    result = st.session_state.qa_chain({"query": query})
                    
                    # Store the response and source documents
                    response = result["result"]
                    sources = result["source_documents"][:3]
                    
                    # Extract source names for citation
                    source_names = [os.path.basename(doc.metadata.get('source', 'Unknown source')) for doc in sources]
                    
                    # Add confidence level calculation (simple example)
                    # This could be enhanced with actual confidence scores from the model
                    similarity_scores = [0.8, 0.7, 0.6]  # Placeholder scores - in production, use actual scores
                    avg_score = sum(similarity_scores) / len(similarity_scores)
                    
                    if avg_score > 0.7:
                        confidence = "high"
                    elif avg_score > 0.5:
                        confidence = "medium"
                    else:
                        confidence = "low"
                    
                    # Append meta information to the response
                    response_with_meta = {
                        "content": response,
                        "sources": source_names,
                        "confidence": confidence
                    }
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_with_meta})
                    
                    # Store source documents for this query
                    st.session_state.last_sources = sources
                    
                    logger.info("Query processed successfully.")
                except Exception as e:
                    error_msg = f"I encountered an error processing your question. Please try again or rephrase your question. Technical details: {str(e)}"
                    logger.error(f"Error processing query: {str(e)}")
                    st.session_state.chat_history.append({"role": "assistant", "content": {"content": error_msg, "confidence": "low", "sources": []}})
            
            # Clear the input box after submitting by forcing a page refresh
            st.rerun()
    
    with col2:
        # Show sources for the last query if available with improved formatting
        if st.session_state.chat_history and hasattr(st.session_state, 'last_sources'):
            st.subheader("Source Documents")
            for i, doc in enumerate(st.session_state.last_sources):
                source = doc.metadata.get('source', 'Unknown source')
                source_name = os.path.basename(source)
                
                with st.expander(f"Source {i+1}: {source_name}"):
                    # Format content preview with better styling
                    st.markdown("**Document Path:**")
                    st.code(source, language="text")
                    
                    st.markdown("**Content Preview:**")
                    content_preview = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                    st.markdown(f"```\n{content_preview}\n```")
                    
                    # Add a "Relevance" score (placeholder)
                    # In a real implementation, you would use the actual similarity score
                    st.markdown(f"**Relevance Score:** {round((0.9 - (i * 0.1)), 2)}")
        
        # Add a help section
        with st.expander("Tips for Better Results"):
            st.markdown("""
            ### How to get better answers from CAIKA:
            
            - **Be specific** in your questions
            - **Use keywords** from your corporate documents
            - **Ask one question** at a time
            - **Review the sources** to understand where information comes from
            - **Upload more documents** to expand CAIKA's knowledge
            """)
        
        # System status and controls 
        st.subheader("Chat Controls")
        if st.button("Clear Conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
            
        if st.button("Regenerate Last Response", key="regenerate"):
            if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
                # Remove the last assistant response
                st.session_state.chat_history.pop()
                # Keep the user query and regenerate
                st.experimental_rerun()
