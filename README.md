# RAG Document Validator Framework

A specialized AI framework for the automatic search, validation, and verification of documentary content. This project demonstrates enterprise-grade RAG architectures, LangGraph workflows, and vector databases.

Udemy Certificate: https://www.udemy.com/certificate/UC-b9fe4c49-5e6f-45a2-ac35-5f4f3751d9f5/

## Overview

In corporate environments (ISO 27001, internal procedures), manually verifying documents is error-prone and time-consuming. This framework leverages **Retrieval-Augmented Generation (RAG)** to:
1.  **Ingest** complex documents (PDF, DOCX, PPTX).
2.  **Validate** specific compliance requirements automatically.
3.  **Chat** with documents using semantic search.
4.  **Verify** answers to mitigate hallucinations (Grounding & Confidence Scoring).

## Key Features

-   **Multi-Format Support**: PDF, Word, PowerPoint.
-   **Pluggable LLMs**: Designed to work with **Llama 3 (Free & Local)** via Ollama, but also supports OpenAI and Mistral.
-   **Vector Database**: Support for Chroma (local), FAISS (local), and Pinecone (cloud).
-   **Hallucination Mitigation**: Built-in `verify()` loop that scores answers based on context evidence.
-   **Compliance Checking**: dedicated `validate()` mode to check "Yes/No" presence of specific clauses.
-   **API & CLI**: Full Flask REST API and Command Line Interface.

## Technology Stack

-   **Core**: Python 3.10+, LangChain
-   **Models**: Llama 3 (Recommended), OpenAI GPT-4o, Mistral
-   **Vector Stores**: ChromaDB, FAISS, Pinecone
-   **API**: Flask

## Prerequisites: Running Llama 3 Locally

This project is optimized to run completely free using **Llama 3** locally.

1.  **Download Ollama**: Visit [ollama.com](https://ollama.com) and install Ollama for your OS.
2.  **Pull the Model**: Open your terminal and run:
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```
    *Note: `nomic-embed-text` is used for high-quality local embeddings.*

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/monaii/Rag-system.git
    cd Rag-system
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Copy `.env.example` to `.env`. If you are using Llama 3 locally as described above, no API keys are needed!
    ```bash
    cp .env.example .env
    ```
    *Ensure `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3` are set in your `.env` file.*

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

(Optional) If you wish to implement evaluation metrics, you can integrate frameworks like Ragas or TruthfulQA.

## Project Structure

```
├── data/               # Raw documents (PDF, DOCX, etc.)
├── docval.py           # Main application logic (CLI + API)
├── RAG-PROJECT.ipynb   # Prototyping and experiments notebook
└── requirements.txt    # Dependencies
```
