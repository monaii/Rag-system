# RAG Document Validation Framework

A lightweight AI system for automatic search, validation, and verification of documentary content. Built for compliance workflows (e.g., ISO 27001) using RAG, vector databases, and LangGraph.

## Author & Training

**Udemy Certificate — RAG Systems with LangChain & Vector Databases**  
[View Certificate](https://www.udemy.com/certificate/UC-b9fe4c49-5e6f-45a2-ac35-5f4f3751d9f5/)

## Features

- **Ingests** PDF, DOCX, PPTX
- **RAG-based question answering** using local Llama 3 via Ollama
- **Vector stores**: Chroma (default), FAISS, Pinecone
- **Answer verification** (supported / partially / unsupported)
- **Requirement validation** (yes/no + justification)
- **LangGraph pipeline**: retrieve → generate → verify → END

## Requirements

Install Ollama and pull the models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/monaii/Rag-system.git
   cd Rag-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Environment Setup (`.env`):
   Create a `.env` file with the following configuration:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3
   EMBED_MODEL=nomic-embed-text
   VECTOR_DB=chroma
   ```

## Usage (Notebook)

All functionality is inside the main notebook: `RAG-PROJECT.ipynb`.

**Example usage in the notebook:**
```python
# Ingest documents from data/ folder
ingest()

# Chat with your documents
chat("What is the CISO responsible for?")

# Verify answers against context
verify(question, answer, context)

# Validate specific requirements
validate(["The policy must define an incident response procedure."])

# Run the LangGraph pipeline
app_rag.invoke({"question": "What controls are defined?"})
```

## Project Structure

- `data/` - Input documents
- `vector/` - Chroma DB storage (ignored in git)
- `RAG-PROJECT.ipynb` - Main project notebook
- `requirements.txt` - Python dependencies
