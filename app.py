import streamlit as st
import os
import json
from docval import chat, verify, validate, ingest

st.set_page_config(page_title="RAG DocValidator", layout="wide")

st.title("AI Document Validator & Search")
st.markdown("""
This tool uses **RAG (Retrieval-Augmented Generation)** to analyze corporate documents, 
validate compliance requirements, and answer questions with hallucination checking.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Check for .env file
    if os.path.exists(".env"):
        st.success(".env file found")
    else:
        st.error(".env file missing")
        
    st.markdown("---")
    st.subheader("Manage Knowledge Base")
    if st.button("Re-Ingest Documents"):
        with st.spinner("Ingesting documents from /data..."):
            try:
                res = ingest()
                st.success(f"Ingested {res['count']} chunks into {res['backend']}!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Tabs for different modes
tab1, tab2 = st.tabs(["Chat & Verify", "Compliance Validator"])

# --- TAB 1: Chat ---
with tab1:
    st.header("Chat with your Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "verification" in message:
                v = message["verification"]
                color = "green" if v["verdict"] == "supported" else "orange" if v["verdict"] == "partially_supported" else "red"
                st.caption(f":{color}[Verdict: {v['verdict']} (Confidence: {v['confidence']})]")
            if "sources" in message:
                with st.expander("View Sources"):
                    for s in message["sources"]:
                        st.markdown(f"- `{s['filename']}`")

    # Chat input
    if prompt := st.chat_input("Ask about the security policy..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(prompt)
                answer = response["answer"]
                context = response["context"]
                docs = response["docs"]
                
                # Verify
                verification = {"verdict": "unsupported", "confidence": 0.0}
                if context:
                    verification = verify(prompt, answer, context)
                
                st.markdown(answer)
                
                color = "green" if verification["verdict"] == "supported" else "orange" if verification["verdict"] == "partially_supported" else "red"
                st.caption(f":{color}[Verdict: {verification['verdict']} (Confidence: {verification['confidence']})]")
                
                with st.expander("View Sources"):
                    for s in docs:
                        st.markdown(f"- `{s['filename']}`")
                        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "verification": verification,
            "sources": docs
        })

# --- TAB 2: Validator ---
with tab2:
    st.header("Automated Compliance Validation")
    st.markdown("Enter requirements (one per line) to check if they are present in the documents.")
    
    default_reqs = "The policy must include an incident response process.\nThe policy must define roles and responsibilities.\nData must be encrypted at rest."
    req_input = st.text_area("Requirements", value=default_reqs, height=150)
    
    if st.button("Validate Requirements"):
        requirements = [r.strip() for r in req_input.split('\n') if r.strip()]
        
        if not requirements:
            st.warning("Please enter at least one requirement.")
        else:
            with st.spinner(f"Validating {len(requirements)} requirements..."):
                results = validate(requirements)
                
                for res in results:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            if res["present"]:
                                st.success("Present")
                            else:
                                st.error("Missing")
                        
                        with col2:
                            st.markdown(f"**{res['requirement']}**")
                            st.info(f"AI Analysis: {res['judgment']}")
                            
                        st.divider()
