"""
FsExplorer Indexer - Docling/langextract-backed document indexing service.

Owns the heavy document parsing and chunking pipeline so the chat-facing
`fs-explorer-api` service never has to import Docling or langextract.
"""

import os

# Load .env file from this package's service root, mirroring the api service.
# Keep this path check cheap because package __init__ runs for every submodule
# import; import python-dotenv only when the file exists.
_env_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".env")
)
if os.path.exists(_env_path):
    from dotenv import load_dotenv

    load_dotenv(_env_path)
