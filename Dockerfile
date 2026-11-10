# Small official Python image (Debian "slim" = fewer extra packages = smaller and safer)
FROM python:3.13-slim

# Security best practice: don't run the app as root. Create a normal user and switch to it.
RUN useradd -m -u 1000 user
USER user

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    EMBED_CACHE_DIR=/home/user/.cache/fastembed \
    RETRIEVER=hybrid \
    CHUNK_SIZE=300
# ^ RETRIEVER and CHUNK_SIZE: set them to the winners of your experiment (Hour 7).

WORKDIR /home/user/app

# 1) Install dependencies first. Docker caches this layer, so rebuilding after a code
#    change is fast (requirements.txt did not change -> no reinstall).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 2) Copy the code and data.
COPY --chown=user . .

# 3) At BUILD time: download the embedding model and embed all chunks once.
#    The app then starts in seconds, even on a tiny free server.
RUN python -m rag_lab.index

# Render tells the app which port to use in the PORT variable; 8000 is the fallback.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag_lab.api:app --host 0.0.0.0 --port ${PORT:-8000}"]