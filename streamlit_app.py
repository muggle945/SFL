"""Streamlit app for RAG-based blog planning system."""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag_system import BlogPlanningRAG

# Page configuration
st.set_page_config(
    page_title="Blog Planning System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'data_file_path' not in st.session_state:
    st.session_state.data_file_path = None


def build_index_with_progress(rag: BlogPlanningRAG, data_file: str):
    """Build index with progress updates."""
    try:
        # Build index (progress messages are handled by the spinner in the UI)
        rag.build_index(data_file=data_file)
        return True
    except Exception as e:
        raise e


def main():
    """Main Streamlit app."""
    
    st.title("RAG-Based Blog Planning System")
    st.markdown("Generate comprehensive blog plans using retrieval-augmented generation with Gemini Flash 2.5")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key status
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("Gemini API Key: Configured")
        else:
            st.warning("Gemini API Key: Not found in .env file")
            st.caption("Please set GEMINI_API_KEY in your .env file")
        
        st.divider()
        
        # Data file selection
        st.subheader("Data Source")
        data_option = st.radio(
            "Select data source",
            ["Use default (medium_data.csv)", "Upload CSV file"],
            index=0
        )
        
        data_file_path = None
        if data_option == "Use default (medium_data.csv)":
            if os.path.exists("medium_data.csv"):
                data_file_path = "medium_data.csv"
                st.success(f"Found: medium_data.csv")
            else:
                st.error("medium_data.csv not found in current directory")
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                data_file_path = temp_path
                st.success(f"Uploaded: {uploaded_file.name}")
        
        st.session_state.data_file_path = data_file_path
        
        st.divider()
        
        # Parameters
        st.subheader("Parameters")
        num_refs = st.slider("Number of reference blogs", 1, 20, 5)
        num_sections = st.slider("Number of sections in plan", 3, 10, 5)
        batch_size = st.slider("Embedding batch size", 1, 8, 8)
        
        st.divider()
        
        # Build index button
        if st.button("Build Index", type="primary", use_container_width=True):
            if data_file_path and os.path.exists(data_file_path):
                try:
                    use_gemini = api_key is not None
                    rag = BlogPlanningRAG(
                        embedding_batch_size=batch_size,
                        use_gemini=use_gemini
                    )
                    
                    with st.spinner("Building index... This may take a few minutes."):
                        build_index_with_progress(rag, data_file_path)
                    
                    st.session_state.rag_system = rag
                    st.session_state.index_built = True
                    st.success("Index built successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building index: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            else:
                st.error("Please select or upload a data file first")
    
    # Main content area
    if not st.session_state.index_built:
        st.info("Please build the index first using the sidebar before generating blog plans.")
        
        if st.session_state.data_file_path:
            st.markdown("### Quick Start")
            st.markdown(f"1. Data file: `{st.session_state.data_file_path}`")
            st.markdown("2. Click 'Build Index' in the sidebar")
            st.markdown("3. Wait for the index to be built")
            st.markdown("4. Enter a topic below to generate a blog plan")
    else:
        st.success("Index is ready! Enter a topic below to generate a blog plan.")
        
        st.divider()
        
        # Topic input
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input(
                "Enter blog topic",
                placeholder="e.g., machine learning, data science, web development",
                key="topic_input"
            )
        
        with col2:
            generate_button = st.button("Generate Plan", type="primary", use_container_width=True)
        
        if generate_button:
            if not topic or len(topic.strip()) == 0:
                st.warning("Please enter a topic")
            else:
                rag = st.session_state.rag_system
                
                with st.spinner("Generating blog plan..."):
                    try:
                        plan = rag.plan_blog(
                            topic=topic,
                            num_references=num_refs,
                            num_sections=num_sections
                        )
                        
                        # Display results
                        st.divider()
                        st.header(f"Blog Plan: {plan['topic']}")
                        
                        # Display generated plan
                        if 'generated_plan' in plan and plan.get('model') != 'fallback':
                            st.subheader("Generated Blog Plan")
                            st.markdown(plan['generated_plan'])
                            
                            if 'suggested_title' in plan:
                                st.info(f"**Suggested Title:** {plan['suggested_title']}")
                        else:
                            st.warning("Gemini generation not available. Showing reference blogs only.")
                        
                        st.divider()
                        
                        # Display reference blogs
                        st.subheader(f"Reference Blogs ({plan['num_references']})")
                        
                        for i, ref in enumerate(plan['references'], 1):
                            with st.expander(f"{i}. {ref['title']}", expanded=(i == 1)):
                                if ref.get('subtitle'):
                                    st.markdown(f"**Subtitle:** {ref['subtitle']}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Reading Time", f"{ref['reading_time']} min")
                                with col2:
                                    st.metric("Claps", ref['claps'])
                                with col3:
                                    st.metric("Similarity", f"{ref['similarity']:.3f}")
                                
                                if ref.get('url'):
                                    st.markdown(f"**URL:** {ref['url']}")
                        
                    except Exception as e:
                        st.error(f"Error generating blog plan: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
    
    # Footer
    st.divider()
    st.caption("RAG-based Blog Planning System | Powered by Gemini Flash 2.5, FAISS, and Sentence Transformers")


if __name__ == "__main__":
    main()

