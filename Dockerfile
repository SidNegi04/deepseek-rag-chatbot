FROM python:3.12-slim

WORKDIR /app

# Needed to fetch and unzip the F1 database at build time
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY rag.py tools.py agent.py server.py ./
COPY docs/ ./docs/
COPY faiss_index/ ./faiss_index/

# Pull the latest F1DB SQLite database (https://github.com/f1db/f1db) so
# query_f1_database has data to query. Re-run the build to pick up a newer
# release; no manual download needed. The archive's exact filename can
# vary between releases, so find whatever .sqlite file comes out and
# normalize it to f1db/f1db.sqlite.
RUN mkdir -p f1db \
    && curl -fL -o /tmp/f1db-sqlite.zip https://github.com/f1db/f1db/releases/latest/download/f1db-sqlite.zip \
    && unzip -o /tmp/f1db-sqlite.zip -d /tmp/f1db-extracted \
    && find /tmp/f1db-extracted \( -name "*.sqlite" -o -name "*.db" \) -exec mv {} f1db/f1db.sqlite \; \
    && test -s f1db/f1db.sqlite \
    && rm -rf /tmp/f1db-sqlite.zip /tmp/f1db-extracted

EXPOSE 8000

CMD ["python3", "server.py"]
