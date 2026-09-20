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
from tools import get_tools, F1DB_PATH, list_databases, DATABASES_DIR
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
_agent_app_had_index = False
_agent_app_db_id = None
_chat_history = []  # simple single-session memory: list of (speaker, text)


def get_agent_app(db_id: str = "f1"):
    global _agent_app, _agent_app_had_index, _agent_app_db_id
    # Rebuild whenever unset, whenever the on-disk index's existence has
    # changed since we last built the agent (see note below), or whenever
    # the selected database has changed since the last build.
    index_now = index_exists()
    if (
        _agent_app is None
        or index_now != _agent_app_had_index
        or db_id != _agent_app_db_id
    ):
        vectorstore = load_vectorstore()
        tools = get_tools(vectorstore, db_id=db_id)
        _agent_app = build_agent(tools)
        _agent_app_had_index = index_now
        _agent_app_db_id = db_id
    return _agent_app


@app.on_event("startup")
def startup_event():
    # Preload the agent (embeddings model, FAISS index, LangGraph agent)
    # at server boot instead of on the first chat request, so the first
    # real user request isn't the one paying for a slow model load.
    get_agent_app()


class ChatRequest(BaseModel):
    message: str
    db_id: str = "f1"


# NOTE: this route is a plain `def`, not `async def`, on purpose.
# Everything inside (agent build, LangChain calls to the hosted LLM,
# embeddings/FAISS work) is synchronous/blocking. In FastAPI, a sync `def`
# route automatically runs in a worker thread pool, off the main event
# loop. If this were `async def`, that blocking work would freeze
# Uvicorn's single event loop entirely, stalling every other request
# (including /api/status) until it finished.
def _looks_truncated(text: str) -> bool:
    """Heuristic: does this look like it was cut off mid-thought?
    Free-tier models sometimes stop early without hitting any token
    cap. A reply that doesn't end in normal closing punctuation is a
    good signal it trailed off (e.g. ends in ':', ',', or a bare word)."""
    stripped = text.rstrip()
    if not stripped:
        return True
    return stripped[-1] not in ".!?\"')]}`"


@app.post("/api/chat")
def chat(request: ChatRequest):
    global _chat_history
    app_graph = get_agent_app(db_id=request.db_id)
    reply, updated_history = run_turn(app_graph, request.message, _chat_history)

    if _looks_truncated(reply):
        # One retry: free-tier models occasionally stop early with no
        # error and no token-cap hit, so a fresh attempt at the same
        # turn is the simplest recovery.
        retry_reply, retry_history = run_turn(app_graph, request.message, _chat_history)
        if not _looks_truncated(retry_reply):
            reply, updated_history = retry_reply, retry_history

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


@app.get("/api/databases")
def databases():
    return {"databases": list_databases()}


@app.post("/api/databases/upload")
async def upload_database(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".sqlite", ".db")):
        return {"error": "Only .sqlite or .db files are supported."}
    DATABASES_DIR.mkdir(exist_ok=True)
    dest = DATABASES_DIR / file.filename
    with open(dest, "wb") as out:
        out.write(await file.read())
    return {"saved": file.filename, "databases": list_databases()}


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