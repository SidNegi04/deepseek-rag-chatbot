"""
server.py - FastAPI backend exposing the LangGraph RAG + tools agent
to the React frontend over HTTP.
"""
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import build_vectorstore, load_vectorstore, index_exists, DOCS_DIR
from tools import get_tools, F1DB_PATH
from agent import build_agent, run_turn

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent_app = None
_chat_history = []  # simple single-session memory: list of (speaker, text)


def get_agent_app():
    global _agent_app
    if _agent_app is None:
        vectorstore = load_vectorstore()
        tools = get_tools(vectorstore)
        _agent_app = build_agent(tools)
    return _agent_app


@app.on_event("startup")
def startup_event():
    # Preload the agent (embeddings model, FAISS index, LangGraph agent)
    # at server boot instead of on the first chat request, so the first
    # real user request isn't the one paying for a slow model load.
    get_agent_app()


class ChatRequest(BaseModel):
    message: str


# NOTE: this route is a plain `def`, not `async def`, on purpose.
# Everything inside (agent build, LangChain calls to the hosted LLM,
# embeddings/FAISS work) is synchronous/blocking. In FastAPI, a sync `def`
# route automatically runs in a worker thread pool, off the main event
# loop. If this were `async def`, that blocking work would freeze
# Uvicorn's single event loop entirely, stalling every other request
# (including /api/status) until it finished.
@app.post("/api/chat")
def chat(request: ChatRequest):
    global _chat_history
    app_graph = get_agent_app()
    reply, updated_history = run_turn(app_graph, request.message, _chat_history)
    _chat_history = updated_history
    return {"response": reply}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    DOCS_DIR.mkdir(exist_ok=True)
    saved = []
    for f in files:
        dest = DOCS_DIR / f.filename
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append(f.filename)
    return {"saved": saved}


# Also sync, for the same reason as /api/chat: build_vectorstore() does
# blocking embedding/FAISS work.
@app.post("/api/reindex")
def reindex():
    global _agent_app
    n_chunks = build_vectorstore()
    _agent_app = None
    return {"chunks_indexed": n_chunks}


@app.post("/api/reset")
async def reset_history():
    global _chat_history
    _chat_history = []
    return {"status": "cleared"}


@app.get("/api/status")
async def status():
    return {
        "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        "index_exists": index_exists(),
        "f1_database_present": F1DB_PATH.exists(),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)