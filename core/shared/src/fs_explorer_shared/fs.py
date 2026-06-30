"""
Basic filesystem utilities shared by the api and indexer services.

Document parsing (Docling) lives in `fs_explorer_indexer.document_parser`, not
here, so that importing this module never pulls Docling into a process.
"""

import os
import re
import glob as glob_module
from pathlib import Path


# =============================================================================
# Configuration Constants
# =============================================================================

# Supported document extensions for parsing
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"}
)


# =============================================================================
# Directory Operations
# =============================================================================


def describe_dir_content(directory: str) -> str:
    """
    Describe the contents of a directory.

    Lists all files and subdirectories in the given directory path.

    Args:
        directory: Path to the directory to describe.

    Returns:
        A formatted string describing the directory contents,
        or an error message if the directory doesn't exist.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"

    children = os.listdir(directory)
    if not children:
        return f"Directory {directory} is empty"

    files = []
    directories = []

    for child in children:
        fullpath = os.path.join(directory, child)
        if os.path.isfile(fullpath):
            files.append(fullpath)
        else:
            directories.append(fullpath)

    description = f"Content of {directory}\n"
    description += "FILES:\n- " + "\n- ".join(files)

    if not directories:
        description += "\nThis folder does not have any sub-folders"
    else:
        description += "\nSUBFOLDERS:\n- " + "\n- ".join(directories)

    return description


# =============================================================================
# Basic File Operations
# =============================================================================


def read_file(file_path: str) -> str:
    """
    Read the contents of a text file.

    Args:
        file_path: Path to the file to read.

    Returns:
        The file contents, or an error message if the file doesn't exist.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    with open(file_path, "r") as f:
        return f.read()


def grep_file_content(file_path: str, pattern: str) -> str:
    """
    Search for a regex pattern in a file.

    Args:
        file_path: Path to the file to search.
        pattern: Regular expression pattern to search for.

    Returns:
        A formatted string with matches, "No matches found",
        or an error message if the file doesn't exist.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    with open(file_path, "r") as f:
        content = f.read()

    regex = re.compile(pattern=pattern, flags=re.MULTILINE)
    matches = regex.findall(content)

    if matches:
        return f"MATCHES for {pattern} in {file_path}:\n\n- " + "\n- ".join(matches)
    return "No matches found"


def glob_paths(directory: str, pattern: str) -> str:
    """
    Find files matching a glob pattern in a directory.

    Args:
        directory: Path to the directory to search in.
        pattern: Glob pattern to match (e.g., "*.txt", "**/*.pdf").

    Returns:
        A formatted string with matching paths, "No matches found",
        or an error message if the directory doesn't exist.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"

    # Use pathlib for cleaner path handling
    search_path = Path(directory) / pattern
    matches = glob_module.glob(str(search_path))

    if matches:
        return f"MATCHES for {pattern} in {directory}:\n\n- " + "\n- ".join(matches)
    return "No matches found"
