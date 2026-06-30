"""Tests for Docling-backed document parsing functions."""

import pytest
import os
from pathlib import Path

from fs_explorer_indexer.document_parser import (
    parse_file,
    preview_file,
    scan_folder,
    clear_document_cache,
)


class TestDocumentParsing:
    """Tests for document parsing functions (parse_file, preview_file)."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_document_cache()

    def test_parse_file_nonexistent(self) -> None:
        """Test parsing a file that doesn't exist."""
        content = parse_file("data/nonexistent.pdf")
        assert content == "No such file: data/nonexistent.pdf"

    def test_parse_file_unsupported_extension(self) -> None:
        """Test parsing a file with unsupported extension."""
        content = parse_file("tests/testfiles/file1.txt")
        assert "Unsupported file extension: .txt" in content

    def test_preview_file_nonexistent(self) -> None:
        """Test previewing a file that doesn't exist."""
        content = preview_file("data/nonexistent.pdf")
        assert content == "No such file: data/nonexistent.pdf"

    def test_preview_file_unsupported_extension(self) -> None:
        """Test previewing a file with unsupported extension."""
        content = preview_file("tests/testfiles/file1.txt")
        assert "Unsupported file extension: .txt" in content

    @pytest.mark.skipif(
        not os.path.exists("data/large_acquisition"),
        reason="Test documents not generated",
    )
    def test_parse_file_pdf(self) -> None:
        """Test parsing an actual PDF file."""
        # Use one of the generated test PDFs
        pdf_files = list(Path("data/large_acquisition").glob("*.pdf"))
        if pdf_files:
            content = parse_file(str(pdf_files[0]))
            assert len(content) > 0
            assert "Error" not in content

    @pytest.mark.skipif(
        not os.path.exists("data/large_acquisition"),
        reason="Test documents not generated",
    )
    def test_preview_file_pdf(self) -> None:
        """Test previewing an actual PDF file."""
        pdf_files = list(Path("data/large_acquisition").glob("*.pdf"))
        if pdf_files:
            content = preview_file(str(pdf_files[0]), max_chars=500)
            assert "=== PREVIEW of" in content
            # Preview should be limited
            assert len(content) < 2000  # Preview + header + truncation message


class TestScanFolder:
    """Tests for scan_folder function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_document_cache()

    def test_nonexistent_directory(self) -> None:
        """Test scanning a directory that doesn't exist."""
        result = scan_folder("nonexistent/path")
        assert result == "No such directory: nonexistent/path"

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning a directory with no supported documents."""
        Path(tmp_path, "test.txt").write_text("hello")
        result = scan_folder(str(tmp_path))
        assert "No supported documents found" in result

    @pytest.mark.skipif(
        not os.path.exists("data/large_acquisition"),
        reason="Test documents not generated",
    )
    def test_scan_folder_with_documents(self) -> None:
        """Test scanning a folder with actual documents."""
        result = scan_folder("data/large_acquisition", max_workers=2)
        assert "PARALLEL DOCUMENT SCAN" in result
        assert "Found" in result
        assert "documents" in result
