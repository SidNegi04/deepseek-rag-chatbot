"""
tools.py - internal document search (with reranking), web search,
calculator, and F1 database tools.
"""
import ast
import operator
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Path to the F1DB SQLite database. Built into the Docker image at build
# time (see Dockerfile) from https://github.com/f1db/f1db releases.
F1DB_PATH = Path(os.environ.get("F1DB_SQLITE_PATH", "f1db/f1db.sqlite"))
DATABASES_DIR = Path(__file__).parent / "databases"


def list_databases() -> list[dict]:
    """Lists all databases available to query: the built-in F1 database
    plus any user-uploaded .sqlite/.db files in DATABASES_DIR."""
    dbs = []
    if F1DB_PATH.exists():
        dbs.append({"id": "f1", "name": "Formula 1 Database", "path": str(F1DB_PATH)})
    if DATABASES_DIR.exists():
        for f in sorted(DATABASES_DIR.iterdir()):
            if f.suffix.lower() in (".sqlite", ".db"):
                dbs.append({"id": f.stem, "name": f.stem, "path": str(f)})
    return dbs

def _make_document_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def run_retrieval(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the internal documents."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[{i}] (source: {source})\n{doc.page_content}")
        return "\n\n".join(results)

    return Tool(
        name="search_internal_documents",
        func=run_retrieval,
        description=(
            "Use this to answer questions about internal/private documents "
            "that have been uploaded and indexed. Input should be a "
            "natural-language search query. Prefer this tool first for "
            "anything that might be covered by the user's own documents."
        ),
    )


_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

# Matches phrasing like "12% of 500" or "12 % of 500"
_PERCENT_OF_RE = re.compile(r"([\d.]+)\s*%\s*of\s*([\d.]+)", re.IGNORECASE)


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression.")


def calculate(expression: str) -> str:
    try:
        # Handle "X% of Y" phrasing before falling back to raw math parsing
        percent_match = _PERCENT_OF_RE.search(expression)
        if percent_match:
            pct = float(percent_match.group(1))
            base = float(percent_match.group(2))
            remainder = expression[percent_match.end():].strip()
            result = (pct / 100) * base
            if remainder:
                # e.g. "12% of 500 + 10" -> compute the rest normally
                full_expr = f"{result}{remainder}"
                tree = ast.parse(full_expr, mode="eval")
                return str(_safe_eval(tree.body))
            return str(result)

        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Could not evaluate expression: {e}"



def get_current_date(_input: str = "") -> str:
    """Returns the actual current date/time from the system clock."""
    return datetime.now().strftime("%A, %B %d, %Y")


def _db_connect(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}.")
    # Read-only connection via URI mode: the agent can only ever read data,
    # never modify the database, no matter what SQL it's tricked into writing.
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _make_schema_tool(db_path: Path, db_name: str):
    def schema(_input: str = "") -> str:
        try:
            conn = _db_connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
                tables = [row[0] for row in cur.fetchall()]
                lines = []
                for table in tables:
                    cur.execute(f"PRAGMA table_info('{table}')")
                    cols = [row[1] for row in cur.fetchall()]
                    lines.append(f"{table}({', '.join(cols)})")
                return "\n".join(lines)
            finally:
                conn.close()
        except Exception as e:
            return f"Could not read {db_name} schema: {e}"
    return schema


_DISALLOWED_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", re.IGNORECASE
)


def _make_query_tool(db_path: Path, db_name: str):
    def query(query: str) -> str:
        stripped = query.strip().rstrip(";")
        if not stripped.lower().startswith("select"):
            return "Only SELECT queries are allowed."
        if _DISALLOWED_SQL_RE.search(stripped):
            return "Query contains a disallowed keyword. Only read-only SELECT queries are allowed."
        if "limit" not in stripped.lower():
            stripped += " LIMIT 50"
        try:
            conn = _db_connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(stripped)
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
                if not rows:
                    return "Query returned no rows."
                lines = [", ".join(columns)]
                lines += [", ".join(str(v) for v in row) for row in rows]
                return "\n".join(lines)
            finally:
                conn.close()
        except Exception as e:
            return f"SQL error: {e}"
    return query


def get_tools(vectorstore=None, db_id: str = "f1"):
    tools = []
    if vectorstore is not None:
        tools.append(_make_document_tool(vectorstore))
    tools.append(DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Use this to look up current or general information from the "
            "public internet. Input should be a search query."
        ),
    ))
    tools.append(Tool(
        name="get_current_date",
        func=get_current_date,
        description=(
            "Use this for ANY question about today's date, the current day, "
            "or 'what day is it'. This gives the exact, guaranteed-accurate "
            "date. Always prefer this over web_search for date questions."
        ),
    ))
    tools.append(Tool(
        name="calculator",
        func=calculate,
        description=(
            "Use this for arithmetic calculations, including percentages. "
            "Examples: '23 * 47 + 1', '12% of 500'."
        ),
    ))

    available = {db["id"]: db for db in list_databases()}
    selected = available.get(db_id) or available.get("f1")
    if selected is not None:
        db_path = Path(selected["path"])
        db_name = selected["name"]
        tools.append(Tool(
            name="database_schema",
            func=_make_schema_tool(db_path, db_name),
            description=(
                f"Use this FIRST, before query_database, whenever you don't "
                f"already know the exact table/column names you need. Returns "
                f"the list of tables and columns in the currently selected "
                f"database ({db_name}). Input is ignored."
            ),
        ))
        tools.append(Tool(
            name="query_database",
            func=_make_query_tool(db_path, db_name),
            description=(
                f"Use this to answer questions that require querying the "
                f"currently selected database ({db_name}). Input must be a "
                f"single read-only SQL SELECT statement. Call database_schema "
                f"first if you're unsure of the table/column names."
            ),
        ))
    return tools
