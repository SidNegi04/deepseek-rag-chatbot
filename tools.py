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


def _f1_connect():
    if not F1DB_PATH.exists():
        raise FileNotFoundError(
            f"F1 database not found at {F1DB_PATH}. It should be downloaded "
            "into the image at build time - see the Dockerfile."
        )
    # Read-only connection via URI mode: the agent can only ever read data,
    # never modify the database, no matter what SQL it's tricked into writing.
    return sqlite3.connect(f"file:{F1DB_PATH}?mode=ro", uri=True)


def f1_schema(_input: str = "") -> str:
    """Lists F1 database tables and their columns, so the agent knows what it can query."""
    try:
        conn = _f1_connect()
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
        return f"Could not read F1 database schema: {e}"


_DISALLOWED_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", re.IGNORECASE
)


def query_f1_database(query: str) -> str:
    """Runs a read-only SQL SELECT query against the F1 database and returns the rows."""
    stripped = query.strip().rstrip(";")
    if not stripped.lower().startswith("select"):
        return "Only SELECT queries are allowed."
    if _DISALLOWED_SQL_RE.search(stripped):
        return "Query contains a disallowed keyword. Only read-only SELECT queries are allowed."
    if "limit" not in stripped.lower():
        stripped += " LIMIT 50"
    try:
        conn = _f1_connect()
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


def get_tools(vectorstore=None):
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
    tools.append(Tool(
        name="f1_database_schema",
        func=f1_schema,
        description=(
            "Use this FIRST, before query_f1_database, whenever you don't already "
            "know the exact table/column names you need. Returns the list of "
            "tables and columns in the Formula 1 database. Input is ignored."
        ),
    ))
    tools.append(Tool(
        name="query_f1_database",
        func=query_f1_database,
        description=(
            "Use this to answer questions about Formula 1 drivers, constructors, "
            "races, results, standings, circuits, or seasons (1950-present). "
            "Input must be a single read-only SQL SELECT statement against the "
            "F1 database. Call f1_database_schema first if you're unsure of the "
            "table/column names."
        ),
    ))
    return tools
