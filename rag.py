"""
rag.py - loads documents from docs/, splits them, embeds them,
and builds/loads a local FAISS vector store.

Embeddings are computed via Hugging Face's free hosted Inference API
instead of loading the sentence-transformers model (and PyTorch) locally.
This keeps the deployed process's memory footprint small enough to fit
Render's free 512MB tier. Requires a free HF_TOKEN env var - get one at
https://huggingface.co/settings/tokens (a "Read" token is enough).
"""
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
    )


def _load_documents():
    documents = []
    if not DOCS_DIR.exists():
        return documents
    for path in DOCS_DIR.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                documents.extend(PyPDFLoader(str(path)).load())
            elif suffix in (".txt", ".md"):
                documents.extend(TextLoader(str(path), encoding="utf-8").load())
        except Exception as e:
            print(f"[rag] Skipped {path.name}: {e}")
    return documents


def build_vectorstore():
    raw_docs = _load_documents()
    if not raw_docs:
        return 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    return len(chunks)


def load_vectorstore():
    if not (INDEX_DIR / "index.faiss").exists():
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def index_exists():
    return (INDEX_DIR / "index.faiss").exists()