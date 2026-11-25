FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pharma_agent/ pharma_agent/
COPY configs/ configs/
COPY eval/questions.jsonl eval/questions.jsonl

# Run as a non-root user; the SQLite run log lives in /app/data (mount a volume to keep it).
RUN useradd --create-home app && mkdir -p data && chown -R app /app
USER app

EXPOSE 8000 8765
# Default: the API. docker-compose also starts the MCP server from this same image.
CMD ["uvicorn", "pharma_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
