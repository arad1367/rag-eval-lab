"""All settings in one place, read from environment variables (or the .env file).

Why: "12-factor app" rule - code stays the same, configuration changes per environment
(your laptop, CI, Render). Secrets never live in code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env if it exists; does nothing otherwise (e.g. in CI)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPERS_PATH = DATA_DIR / "papers.jsonl"
EVAL_PATH = DATA_DIR / "eval_questions.jsonl"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_cache_dir: str = os.getenv("EMBED_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "fastembed"))
    retriever: str = os.getenv("RETRIEVER", "hybrid")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))


settings = Settings()
