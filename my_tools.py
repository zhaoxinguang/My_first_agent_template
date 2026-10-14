from pathlib import Path
from smolagents import tool


# 只允许读取这个目录下的文件
BASE_DIR = Path(__file__).resolve().parent / "documents"


@tool
def read_text_file(
    filename: str,
    start_line: int = 1,
    end_line: int = -1,
) -> str:
    """Read all or part of a local UTF-8 text file.

    Args:
        filename: Name of the .txt file inside the documents directory,
            for example notes.txt. Do not provide an absolute path.
        start_line: First line to read, starting from 1. Defaults to 1.
        end_line: Last line to read, inclusive. Use -1 to read to the end.
            Defaults to -1.
    """
    if not filename.endswith(".txt"):
        return "Error: Only .txt files are allowed."

    if start_line < 1:
        return "Error: start_line must be at least 1."

    if end_line != -1 and end_line < start_line:
        return "Error: end_line must be -1 or greater than or equal to start_line."

    base_dir = BASE_DIR.resolve()
    file_path = (base_dir / filename).resolve()

    # Prevent path traversal, such as ../../secret.txt
    if file_path.parent != base_dir:
        return "Error: Access outside the documents directory is not allowed."

    if not file_path.is_file():
        return f"Error: File not found: {filename}"

    try:
        with file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line > len(lines):
            return f"Error: File has only {len(lines)} lines."

        last = len(lines) if end_line == -1 else min(end_line, len(lines))
        content = "".join(lines[start_line - 1:last])

        return (
            f"File: {filename}\n"
            f"Lines: {start_line}-{last}\n"
            f"Content:\n{content}"
        )

    except UnicodeDecodeError:
        return "Error: The file is not valid UTF-8 text."
    except OSError as e:
        return f"Error reading file: {e}"
    
    
@tool
def list_files() -> str:
    """List available .txt files in the documents directory.

    Use this tool before reading a file when you do not know
    which filenames are available. Returns the available filenames.
    """
    base_dir = BASE_DIR.resolve()

    if not base_dir.is_dir():
        return "Error: Documents directory does not exist."

    try:
        files = sorted(
            p.name
            for p in base_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        )

        if not files:
            return "No .txt files are available."

        return "Available text files:\n" + "\n".join(files)

    except OSError as e:
        return f"Error listing files: {e}"

