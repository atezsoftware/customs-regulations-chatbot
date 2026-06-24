# Agentic File Search

> **Based on**: [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer) — The original CLI agent for filesystem exploration.

An AI-powered document search agent that explores files like a human would — scanning, reasoning, and following cross-references. Unlike traditional RAG systems that rely on pre-computed embeddings, this agent dynamically navigates documents to find answers.

## Why Agentic Search?

Traditional RAG (Retrieval-Augmented Generation) has limitations:
- **Chunks lose context** — Splitting documents destroys relationships between sections
- **Cross-references are invisible** — "See Exhibit B" means nothing to embeddings
- **Similarity ≠ Relevance** — Semantic matching misses logical connections

This system uses a **three-phase strategy**:
1. **Parallel Scan** — Preview all documents in a folder at once
2. **Deep Dive** — Full extraction on relevant documents only
3. **Backtrack** — Follow cross-references to previously skipped documents

## Watch the video
This video explains the architecture of the project and how to run it. 
[![Watch the demo on YouTube](https://img.youtube.com/vi/rMADSuus6jg/maxresdefault.jpg)](https://www.youtube.com/watch?v=rMADSuus6jg)

## Features

- 🔍 **6 Tools**: `scan_folder`, `preview_file`, `parse_file`, `read`, `grep`, `glob`
- 📄 **Document Support**: PDF, DOCX, PPTX, XLSX, HTML, Markdown (via Docling)
- 🤖 **Powered by**: Google Gemini 3 Flash with structured JSON output
- 💰 **Cost Efficient**: ~$0.001 per query with token tracking
- 🌐 **Web UI**: Real-time WebSocket streaming interface
- 📊 **Citations**: Answers include source references

## Installation

```bash
# Clone the repository
git clone https://github.com/PromtEngineer/agentic-file-search.git
cd agentic-file-search

# Create the uv-managed virtual environment
uv venv
source .venv/bin/activate

# Install the app and runtime dependencies
uv pip install .
```

Windows PowerShell:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install .
```

## Configuration

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

### CLI

```bash
# Basic query
uv run explore --task "What is the purchase price in data/test_acquisition/?"

# Multi-document query
uv run explore --task "Look in data/large_acquisition/. What are all the financial terms including adjustments and escrow?"
```

### Web UI

```bash
# Start the server
PYTHONUTF8=1 uv run uvicorn fs_explorer.server:app --host 127.0.0.1 --port 8000

# Or use the installed console script
uv run explore-ui

# Open http://127.0.0.1:8000 in your browser
```

The web UI provides:
- Folder browser to select target directory
- Chat-style interface with temporary in-browser conversation memory
- Real-time status updates for thinking, searching, reading, and analysis
- Streaming final answers with citations
- Token usage and cost statistics
- UTF-8 text handling for Turkish characters and other non-ASCII content

### Turkish / UTF-8 text

The web UI serves UTF-8 HTML and supports Turkish characters in questions,
answers, folder names, and document text. If your terminal or operating system
does not default to UTF-8, start the server with UTF-8 enabled:

```bash
PYTHONUTF8=1 uv run uvicorn fs_explorer.server:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
$env:PYTHONUTF8="1"
uv run uvicorn fs_explorer.server:app --host 127.0.0.1 --port 8000
```

### Optional Indexing

You can ask questions without preparing an index. In that mode, the agent uses
the original file-exploration flow: it scans folders, previews files, and reads
only the sources it decides are relevant.

Indexing is optional. It parses the selected folder once, writes searchable
chunks to DuckDB, and enables the Semantic toggle in the web UI:

```bash
uv run explore index data/customs_test
uv run explore --use-index --task "Transit sure asimi cezasi hangi genelgede geciyor?"
```

## Architecture

```
User Query
    ↓
┌─────────────────┐
│ Workflow Engine │ ←→ LlamaIndex Workflows (event-driven)
└────────┬────────┘
         ↓
┌─────────────────┐
│     Agent       │ ←→ Gemini 3 Flash (structured JSON)
└────────┬────────┘
         ↓
┌─────────────────────────────────────────┐
│ scan_folder │ preview │ parse │ read │ grep │ glob │
└─────────────────────────────────────────┘
                    ↓
              Document Parser (Docling - local)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.

## Test Documents

The repo includes test document sets for evaluation:

- `data/test_acquisition/` — 10 interconnected legal documents
- `data/large_acquisition/` — 25 documents with extensive cross-references

Example queries:
```bash
# Simple (single doc)
uv run explore --task "Look in data/test_acquisition/. Who is the CTO?"

# Cross-reference required
uv run explore --task "Look in data/test_acquisition/. What is the adjusted purchase price?"

# Multi-document synthesis
uv run explore --task "Look in data/large_acquisition/. What happens to employees after the acquisition?"
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini 3 Flash |
| Document Parsing | Docling (local, open-source) |
| Orchestration | LlamaIndex Workflows |
| CLI | Typer + Rich |
| Web Server | FastAPI + WebSocket |
| Package Manager | uv |

## Project Structure

```
src/fs_explorer/
├── agent.py      # Gemini client, token tracking
├── workflow.py   # LlamaIndex workflow engine
├── fs.py         # File tools: scan, parse, grep
├── models.py     # Pydantic models for actions
├── main.py       # CLI entry point
├── server.py     # FastAPI + WebSocket server
└── ui.html       # Single-file web interface
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## License

MIT

## Acknowledgments

- Original concept from [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer)
- Document parsing by [Docling](https://github.com/DS4SD/docling)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/agentic-file-search&type=Date)](https://star-history.com/#PromtEngineer/agentic-file-search&Date)





  /home/kubilay-payci/customs-regulations-chatbot/.venv/bin/python -m fs_explorer.chunk_inspector --host 127.0.0.1 --port 8123