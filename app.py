import streamlit as st
from rag_pipeline import retrieve_relevant_chunks, query_llm_groq
from PIL import Image

# UI Configuration
st.set_page_config(page_title="KMIT RAG Chatbot", layout="wide")

# Sidebar with KMIT branding
with st.sidebar:
    st.image("https://www.kmit.in/images/kmit-bar.png", use_container_width=True)
    st.title("KMIT RAG Chatbot")
    st.markdown("""
        Ask any question based on the content available on KMIT's official website.

        This chatbot is powered by Retrieval-Augmented Generation (RAG).
    """)
    st.markdown("---")
    st.markdown("Developed using Streamlit, FAISS, and Groq API.")

# Main Header
st.markdown("""
    <h1 style='text-align: center;'>📄 KMIT Website Assistant</h1>
    <p style='text-align: center; font-size: 18px;'>Instantly get answers sourced from the KMIT website.</p>
""", unsafe_allow_html=True)

# Search Bar
st.markdown("### 🔍 Ask a question")
query = st.text_input("", placeholder="e.g., What courses does KMIT offer?", label_visibility="collapsed")

# Results Section
if query:
    with st.spinner("🔄 Retrieving relevant content and generating answer..."):
        results = retrieve_relevant_chunks(query)
        context = "\n".join([r[0] for r in results])
        prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
        response = query_llm_groq(prompt, model="llama3-70b-8192", api_key="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")

    st.markdown("---")
    st.markdown("### 💬 Answer")
    st.success(response)

    with st.expander("📚 Sources from KMIT Website"):
        for i, (chunk, meta) in enumerate(results):
            st.markdown(f"{i+1}. [View Source]({meta['source']})")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 14px;'>
        Made with ❤️ by your AI Assistant &nbsp;|&nbsp; Powered by <strong>FAISS</strong> + <strong>Groq</strong> + <strong>Streamlit</strong>
    </div>
""", unsafe_allow_html=True)
