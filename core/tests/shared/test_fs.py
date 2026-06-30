"""Tests for basic filesystem utility functions (no Docling)."""

from fs_explorer_shared.fs import (
    describe_dir_content,
    read_file,
    grep_file_content,
    glob_paths,
    SUPPORTED_EXTENSIONS,
)


class TestDescribeDirContent:
    """Tests for describe_dir_content function."""

    def test_valid_directory(self) -> None:
        """Test describing a valid directory with files and subfolders."""
        description = describe_dir_content("tests/testfiles")
        assert "Content of tests/testfiles" in description
        assert "tests/testfiles/file1.txt" in description
        assert "tests/testfiles/file2.md" in description
        assert "tests/testfiles/last" in description

    def test_nonexistent_directory(self) -> None:
        """Test describing a directory that doesn't exist."""
        description = describe_dir_content("tests/testfile")
        assert description == "No such directory: tests/testfile"

    def test_directory_without_subfolders(self) -> None:
        """Test describing a directory that has no subdirectories."""
        description = describe_dir_content("tests/testfiles/last")
        assert "Content of tests/testfiles/last" in description
        assert "tests/testfiles/last/lastfile.txt" in description
        assert "This folder does not have any sub-folders" in description


class TestReadFile:
    """Tests for read_file function."""

    def test_valid_file(self) -> None:
        """Test reading a valid text file."""
        content = read_file("tests/testfiles/file1.txt")
        assert content.strip() == "this is a test"

    def test_nonexistent_file(self) -> None:
        """Test reading a file that doesn't exist."""
        content = read_file("tests/testfiles/file2.txt")
        assert content == "No such file: tests/testfiles/file2.txt"


class TestGrepFileContent:
    """Tests for grep_file_content function."""

    def test_pattern_match(self) -> None:
        """Test searching for a pattern that exists."""
        result = grep_file_content("tests/testfiles/file2.md", r"(are|is) a test")
        assert "MATCHES for (are|is) a test" in result
        assert "is" in result

    def test_no_match(self) -> None:
        """Test searching for a pattern that doesn't exist."""
        result = grep_file_content("tests/testfiles/last/lastfile.txt", r"test")
        assert result == "No matches found"

    def test_nonexistent_file(self) -> None:
        """Test searching in a file that doesn't exist."""
        result = grep_file_content("tests/testfiles/file2.txt", r"test")
        assert result == "No such file: tests/testfiles/file2.txt"


class TestGlobPaths:
    """Tests for glob_paths function."""

    def test_pattern_match(self) -> None:
        """Test finding files that match a glob pattern."""
        result = glob_paths("tests/testfiles", "file?.*")
        assert "MATCHES for file?.* in tests/testfiles" in result
        assert "file1.txt" in result
        assert "file2.md" in result

    def test_no_match(self) -> None:
        """Test a pattern that matches nothing."""
        result = glob_paths("tests/testfiles", "nonexistent*")
        assert result == "No matches found"

    def test_nonexistent_directory(self) -> None:
        """Test glob in a directory that doesn't exist."""
        result = glob_paths("tests/nonexistent", "*.txt")
        assert result == "No such directory: tests/nonexistent"


class TestSupportedExtensions:
    """Tests for supported extensions configuration."""

    def test_supported_extensions_is_frozenset(self) -> None:
        """Verify SUPPORTED_EXTENSIONS is immutable."""
        assert isinstance(SUPPORTED_EXTENSIONS, frozenset)

    def test_common_extensions_supported(self) -> None:
        """Verify common document extensions are supported."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
