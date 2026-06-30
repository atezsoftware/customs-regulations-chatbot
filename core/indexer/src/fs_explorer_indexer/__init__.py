"""
FsExplorer Indexer - Docling/langextract-backed document indexing service.

Owns the heavy document parsing and chunking pipeline so the chat-facing
`fs-explorer-api` service never has to import Docling or langextract.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env file from this package's service root, mirroring the api service.
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
