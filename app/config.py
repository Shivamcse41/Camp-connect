"""
app/config.py
─────────────
Central configuration for CampConnect SaaS.
All settings are loaded from environment variables (or .env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level up from this file)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Settings:
    # ── Project Root ─────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ── Vector Store ──────────────────────────────────────────────────────────
    # Per-client FAISS indexes are stored at:
    #   vectorstore/{client_id}/db_faiss/
    VECTORSTORE_BASE: Path = BASE_DIR / "vectorstore"

    # ── Embedding Model ───────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Groq LLM ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"

    # ── MySQL Database ────────────────────────────────────────────────────────
    # Format: mysql+pymysql://user:password@host:port/dbname
    MYSQL_URL: str = ""

    def __init__(self):
        # Resolve MYSQL_URL dynamically
        mysql_host = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST")
        mysql_port = os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or "3306"
        mysql_user = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER")
        mysql_pass = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD")
        mysql_db = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE")

        if mysql_host and mysql_user and mysql_db:
            # Construct from individual environment variables
            self.MYSQL_URL = f"mysql+pymysql://{mysql_user}:{mysql_pass or ''}@{mysql_host}:{mysql_port}/{mysql_db}"
        else:
            url = os.getenv("MYSQL_URL") or os.getenv("DATABASE_URL")
            if url:
                if url.startswith("mysql://"):
                    url = url.replace("mysql://", "mysql+pymysql://", 1)
                self.MYSQL_URL = url
            else:
                raise RuntimeError(
                    "No MySQL database configured. Set MYSQL_URL (or DATABASE_URL) as an environment variable, "
                    "or provide individual MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE variables. "
                    "See .env.example for details."
                )

    # ── JWT Auth (used in Task 2) ─────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production-please")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # ── RAG Retrieval ─────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 3
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50


# Single shared settings instance — import this everywhere
settings = Settings()
