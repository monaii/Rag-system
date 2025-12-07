# RAG Document Validator Framework

A specialized AI framework for the automatic search, validation, and verification of documentary content. This project was developed while completing the Udemy course “RAG for Professionals with LangGraph, Python and OpenAI”,
where I studied enterprise-grade RAG architectures, LangGraph workflows, vector databases, and evaluation techniques.
Certificate: https://www.udemy.com/certificate/UC-b9fe4c49-5e6f-45a2-ac35-5f4f3751d9f5/
## Overview

In corporate environments (ISO 27001, internal procedures), manually verifying documents is error-prone and time-consuming. This framework leverages **Retrieval-Augmented Generation (RAG)** to:
1.  **Ingest** complex documents (PDF, DOCX, PPTX).
2.  **Validate** specific compliance requirements automatically.
3.  **Chat** with documents using semantic search.
4.  **Verify** answers to mitigate hallucinations (Grounding & Confidence Scoring).

## Key Features

-   **Multi-Format Support**: PDF, Word, PowerPoint.
-   **Pluggable LLMs**: Supports OpenAI, Mistral, and local Llama 3 (via Ollama).
-   **Vector Database**: Support for Chroma (local), FAISS (local), and Pinecone (cloud).
-   **Hallucination Mitigation**: Built-in `verify()` loop that scores answers based on context evidence.
-   **Compliance Checking**: dedicated `validate()` mode to check "Yes/No" presence of specific clauses.
-   **Evaluation**: Integration with `ragas` for performance metrics.
-   **API & CLI**: Full Flask REST API and Command Line Interface.

## Technology Stack

-   **Core**: Python 3.10+, LangChain
-   **Models**: OpenAI GPT-4o, Mistral, Llama 3
-   **Vector Stores**: ChromaDB, FAISS, Pinecone
-   **API**: Flask
-   **Evaluation**: Ragas, TruthfulQA

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/rag-doc-validator.git
    cd rag-doc-validator
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Copy `.env.example` to `.env` and set your API keys (or use Ollama for free local inference).
    ```bash
    cp .env.example .env
    ```

## Usage

### 1. Ingest Documents
Place your files in the `data/` directory and run:
```bash
python docval.py ingest
```

### 2. Chat with Verification
Ask questions and get grounded answers with confidence scores:
```bash
python docval.py chat "What are the roles defined in the security policy?"
```

### 3. Validate Requirements
Check for specific compliance clauses:
```bash
python docval.py validate "The policy must include an incident response process." "Data must be encrypted at rest."
```

### 4. Run API Server
Start the REST API:
```bash
python docval.py serve
```

## Evaluation

Run Ragas evaluation metrics:
```bash
python eval.py ragas test_questions.json
```

## Project Structure

```
├── data/               # Raw documents (PDF, DOCX, etc.)
├── vector/             # Persistent vector database storage
├── docval.py           # Main application logic (CLI + API)
├── eval.py             # Evaluation scripts (Ragas, Hallucination Index)
├── RAG-PROJECT.ipynb   # Prototyping and experiments notebook
└── requirements.txt    # Dependencies
```
