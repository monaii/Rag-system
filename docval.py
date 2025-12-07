import os
import argparse
import json
from datetime import datetime

from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI
from langchain_community.chat_models import ChatOllama

from flask import Flask, request, jsonify


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DIR = os.path.join(BASE_DIR, "vector")

load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_chat_model():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    if provider == "openai":
        return init_chat_model(model=model, temperature=0, model_provider="openai")
    if provider == "mistral":
        return ChatMistralAI(model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"), temperature=0)
    if provider == "ollama":
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=0)
    return init_chat_model(model=model, temperature=0, model_provider="openai")


def get_retriever():
    emb_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=emb_model)
    backend = os.getenv("VECTOR_DB", "chroma").lower()
    if backend == "chroma":
        persist_dir = os.path.join(VECTOR_DIR, "chroma")
        vectordb = Chroma(collection_name=os.getenv("COLLECTION", "docval"), embedding_function=embeddings, persist_directory=persist_dir)
        return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5})
    if backend == "faiss":
        faiss_dir = os.path.join(VECTOR_DIR, "faiss")
        if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
            vectordb = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            vectordb = FAISS.from_texts([""], embeddings)
            os.makedirs(faiss_dir, exist_ok=True)
            vectordb.save_local(faiss_dir)
        return vectordb.as_retriever(search_kwargs={"k": 8})
    if backend == "pinecone":
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index_name = os.getenv("PINECONE_INDEX", os.getenv("COLLECTION", "docval"))
        if index_name not in [i.name for i in pc.list_indexes()]:
            pc.create_index(name=index_name, dimension=1536, metric="cosine")
        index = pc.Index(index_name)
        vectordb = PineconeVectorStore(index=index, embedding=embeddings, namespace=os.getenv("PINECONE_NAMESPACE", "default"))
        return vectordb.as_retriever(search_kwargs={"k": 8})
    persist_dir = os.path.join(VECTOR_DIR, "chroma")
    vectordb = Chroma(collection_name=os.getenv("COLLECTION", "docval"), embedding_function=embeddings, persist_directory=persist_dir)
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5})


def _load_text(abs_path: str, ftype: str):
    if ftype == "docx":
        return Docx2txtLoader(abs_path).load()[0].page_content
    if ftype == "pdf":
        return PyPDFLoader(abs_path, mode="single").load()[0].page_content
    if ftype == "pptx":
        slides = UnstructuredPowerPointLoader(abs_path).load()
        return "\n\n".join(d.page_content or "" for d in slides)
    raise ValueError("unsupported file type")


def _file_type(path: str):
    ext = os.path.splitext(path)[1].lower()
    return "docx" if ext == ".docx" else "pdf" if ext == ".pdf" else "pptx" if ext == ".pptx" else "other"


def _chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, add_start_index=True)
    docs = splitter.split_documents([Document(page_content=text)])
    return [Document(page_content=d.page_content, metadata={"start_index": d.metadata.get("start_index")}) for d in docs]


def ingest():
    os.makedirs(DATA_DIR, exist_ok=True)
    backend = os.getenv("VECTOR_DB", "chroma").lower()
    emb_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=emb_model)
    collection = os.getenv("COLLECTION", "docval")
    if backend == "chroma":
        persist_dir = os.path.join(VECTOR_DIR, "chroma")
        os.makedirs(persist_dir, exist_ok=True)
        vectordb = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=persist_dir)
        all_docs = []
        for root, _, files in os.walk(DATA_DIR):
            for name in files:
                if name.startswith("~$"):
                    continue
                ftype = _file_type(name)
                if ftype == "other":
                    continue
                p = os.path.join(root, name)
                text = _load_text(p, ftype)
                chunks = _chunk_text(text)
                for d in chunks:
                    md = {"source_path": p.replace("\\", "/"), "filename": name, "type": ftype, "ingested_at": datetime.utcnow().isoformat() + "Z"}
                    all_docs.append(Document(page_content=d.page_content, metadata=md))
        if all_docs:
            vectordb.add_documents(all_docs)
        return {"backend": backend, "count": len(all_docs)}
    if backend == "faiss":
        faiss_dir = os.path.join(VECTOR_DIR, "faiss")
        texts, metas = [], []
        for root, _, files in os.walk(DATA_DIR):
            for name in files:
                ftype = _file_type(name)
                if ftype == "other":
                    continue
                p = os.path.join(root, name)
                text = _load_text(p, ftype)
                for d in _chunk_text(text):
                    texts.append(d.page_content)
                    metas.append({"source_path": p.replace("\\", "/"), "filename": name, "type": ftype})
        vectordb = FAISS.from_texts(texts, embeddings, metadatas=metas) if texts else FAISS.from_texts([""], embeddings)
        os.makedirs(faiss_dir, exist_ok=True)
        vectordb.save_local(faiss_dir)
        return {"backend": backend, "count": len(texts)}
    if backend == "pinecone":
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index_name = os.getenv("PINECONE_INDEX", collection)
        if index_name not in [i.name for i in pc.list_indexes()]:
            pc.create_index(name=index_name, dimension=1536, metric="cosine")
        index = pc.Index(index_name)
        vectordb = PineconeVectorStore(index=index, embedding=embeddings, namespace=os.getenv("PINECONE_NAMESPACE", "default"))
        docs = []
        for root, _, files in os.walk(DATA_DIR):
            for name in files:
                ftype = _file_type(name)
                if ftype == "other":
                    continue
                p = os.path.join(root, name)
                text = _load_text(p, ftype)
                for d in _chunk_text(text):
                    docs.append(Document(page_content=d.page_content, metadata={"source_path": p.replace("\\", "/"), "filename": name, "type": ftype}))
        if docs:
            vectordb.add_documents(docs)
        return {"backend": backend, "count": len(docs)}
    return {"backend": backend, "count": 0}


def _concat(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


def chat(question: str):
    retriever = get_retriever()
    llm = get_chat_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the provided context. If insufficient, say you don't know."),
        MessagesPlaceholder(variable_name="history", n_messages=20),
        ("system", "Context:\n{context}"),
        MessagesPlaceholder(variable_name="question", n_messages=1),
    ])
    chain = prompt | llm
    docs = retriever.invoke(question)
    context = _concat(docs)
    ai = chain.invoke({"history": [], "question": [HumanMessage(content=question)], "context": context})
    return {"answer": ai.content, "context": context, "docs": [{"source": d.metadata.get("source_path"), "filename": d.metadata.get("filename")} for d in docs]}


def verify(question: str, answer: str, context: str):
    llm = get_chat_model()
    vprompt = ChatPromptTemplate.from_messages([
        ("system", "Check if the answer is supported by the context and return one of: supported, partially_supported, unsupported."),
        ("system", "Question:\n{q}\nAnswer:\n{a}\nContext:\n{c}"),
    ])
    vchain = vprompt | llm
    verdict = vchain.invoke({"q": question, "a": answer, "c": context}).content.strip().lower()
    score = 1.0 if verdict == "supported" else 0.5 if verdict == "partially_supported" else 0.0
    return {"verdict": verdict, "confidence": score}


def validate(requirements):
    retriever = get_retriever()
    llm = get_chat_model()
    classifier = ChatPromptTemplate.from_messages([
        ("system", "Decide if the requirement is present in the context. Answer yes or no and provide a short justification."),
        ("system", "Requirement:\n{req}\nContext:\n{ctx}"),
    ])
    out = []
    for req in requirements:
        docs = retriever.invoke(req)
        ctx = _concat(docs)
        ans = (classifier | llm).invoke({"req": req, "ctx": ctx}).content.strip()
        present = ans.lower().startswith("yes")
        out.append({"requirement": req, "present": present, "judgment": ans, "docs": [{"source": d.metadata.get("source_path"), "filename": d.metadata.get("filename")} for d in docs]})
    return out


def create_app():
    app = Flask(__name__)

    @app.post("/ingest")
    def _ingest():
        return jsonify(ingest())

    @app.post("/chat")
    def _chat():
        data = request.get_json(force=True)
        q = data.get("question", "")
        r = chat(q)
        v = verify(q, r["answer"], r["context"]) if r["context"] else {"verdict": "unsupported", "confidence": 0.0}
        return jsonify({"answer": r["answer"], "verdict": v["verdict"], "confidence": v["confidence"], "docs": r["docs"]})

    @app.post("/validate")
    def _validate():
        data = request.get_json(force=True)
        reqs = data.get("requirements", [])
        return jsonify({"results": validate(reqs)})

    return app


def main():
    parser = argparse.ArgumentParser(prog="docval")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ingest")
    p_chat = sub.add_parser("chat")
    p_chat.add_argument("question")
    p_val = sub.add_parser("validate")
    p_val.add_argument("requirements", nargs="+")
    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if args.cmd == "ingest":
        res = ingest()
        print(json.dumps(res, indent=2))
    elif args.cmd == "chat":
        r = chat(args.question)
        v = verify(args.question, r["answer"], r["context"]) if r["context"] else {"verdict": "unsupported", "confidence": 0.0}
        print(json.dumps({"answer": r["answer"], "verdict": v["verdict"], "confidence": v["confidence"], "docs": r["docs"]}, indent=2))
    elif args.cmd == "validate":
        res = validate(args.requirements)
        print(json.dumps({"results": res}, indent=2))
    elif args.cmd == "serve":
        app = create_app()
        app.run(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

