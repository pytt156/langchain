import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.vectorstores import FAISS

from util.tool_config import (
    DOCS_PATH,
    DEFAULT_PROJECT_ROOT,
    MAX_HEADINGS,
    MAX_PARAGRAPHS,
    MAX_FILE_LIST,
    MAX_SEARCH_RESULTS,
    CHUNK_SIZE,
    CHUNK_OVERLAP_LINES,
    ALLOWED_TEXT_EXTENSIONS,
    EXCLUDED_DIR_NAMES,
    EXCLUDED_FILE_NAMES,
)

from util.embeddings import get_embeddings
from util.tool_schema import (
    CalculateInput,
    FetchSummarizeAndSaveInput,
    IndexProjectInput,
    ListFilesInput,
    ReadFileInput,
    ReplaceTextInput,
    SearchCodebaseInput,
    SearchDocumentsInput,
    WriteFileInput,
)

_docs_vectorstore = None
_project_vectorstore = None
_indexed_project_root = None
_indexed_project_file_count = 0


def _ensure_docs_path() -> Path:
    """Ensure the docs directory exists and return it."""
    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    return DOCS_PATH


def _resolve_project_root(root_dir: str | None) -> Path:
    """Resolve the project root, falling back to DEFAULT_PROJECT_ROOT."""
    if root_dir is None:
        return DEFAULT_PROJECT_ROOT
    return Path(root_dir).expanduser().resolve()


def _safe_filename(text: str, max_length: int = 60) -> str:
    """Convert arbitrary text into a filesystem-safe filename stem."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return cleaned[:max_length] or "notes"


def _extract_webpage_notes(html: str, url: str) -> str | None:
    """Extract title, headings, and paragraphs from HTML and format as notes."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = (
        soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
    )

    headings: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        if text:
            headings.append(text)

    paragraphs: list[str] = []
    for tag in soup.find_all("p"):
        text = tag.get_text(" ", strip=True)
        if text and len(text) > 40:
            paragraphs.append(text)

    if not headings and not paragraphs:
        return None

    notes_parts = [
        f"Title: {title}",
        f"URL: {url}",
        "",
        "Headings:",
    ]

    if headings:
        notes_parts.extend(f"- {heading}" for heading in headings[:MAX_HEADINGS])
    else:
        notes_parts.append("- No headings found")

    notes_parts.extend(["", "Summary notes:"])
    if paragraphs:
        notes_parts.extend(paragraphs[:MAX_PARAGRAPHS])
    else:
        notes_parts.append("No paragraph text found")

    return "\n".join(notes_parts).strip()


def _is_allowed_text_file(path: Path) -> bool:
    """Return True if the file looks like a supported text/code file."""
    if not path.is_file():
        return False

    if path.name in EXCLUDED_FILE_NAMES:
        return False

    return path.suffix.lower() in ALLOWED_TEXT_EXTENSIONS


def _iter_project_files(root_dir: Path):
    """Yield project files recursively, excluding common junk/build folders."""
    for path in root_dir.rglob("*"):
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if _is_allowed_text_file(path):
            yield path


def _chunk_text_with_metadata(file_path: Path, text: str) -> list[Document]:
    """Split file text into chunks and attach file metadata to each chunk."""
    lines = text.splitlines()
    documents: list[Document] = []

    if not lines:
        return documents

    chunk_lines: list[str] = []
    chunk_char_count = 0
    chunk_start_line = 1
    chunk_index = 0

    for line_number, line in enumerate(lines, start=1):
        line_length = len(line) + 1

        if chunk_lines and chunk_char_count + line_length > CHUNK_SIZE:
            chunk_text = "\n".join(chunk_lines).strip()
            if chunk_text:
                documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": file_path.suffix.lower(),
                            "chunk_index": chunk_index,
                            "start_line": chunk_start_line,
                            "end_line": line_number - 1,
                        },
                    )
                )
                chunk_index += 1

            overlap = (
                chunk_lines[-CHUNK_OVERLAP_LINES:] if CHUNK_OVERLAP_LINES > 0 else []
            )
            chunk_lines = overlap.copy()
            chunk_char_count = sum(len(x) + 1 for x in chunk_lines)
            chunk_start_line = max(1, line_number - len(chunk_lines))

        chunk_lines.append(line)
        chunk_char_count += line_length

    if chunk_lines:
        chunk_text = "\n".join(chunk_lines).strip()
        if chunk_text:
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "chunk_index": chunk_index,
                        "start_line": chunk_start_line,
                        "end_line": len(lines),
                    },
                )
            )

    return documents


def _invalidate_doc_index() -> None:
    """Clear cached docs vectorstore."""
    global _docs_vectorstore
    _docs_vectorstore = None


def _invalidate_project_index() -> None:
    """Clear cached project vectorstore."""
    global _project_vectorstore, _indexed_project_root, _indexed_project_file_count
    _project_vectorstore = None
    _indexed_project_root = None
    _indexed_project_file_count = 0


def _build_docs_vectorstore():
    """Build a FAISS vectorstore from all .txt files in DOCS_PATH."""
    docs_path = _ensure_docs_path()
    documents: list[Document] = []

    for file_path in docs_path.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            continue

        if not text:
            continue

        documents.extend(_chunk_text_with_metadata(file_path, text))

    if not documents:
        return None

    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def _get_docs_vectorstore():
    """Return cached docs vectorstore, building it if needed."""
    global _docs_vectorstore

    if _docs_vectorstore is None:
        _docs_vectorstore = _build_docs_vectorstore()

    return _docs_vectorstore


def _build_project_vectorstore(root_dir: Path):
    """Build a FAISS vectorstore from project files."""
    documents: list[Document] = []
    file_count = 0

    for file_path in _iter_project_files(root_dir):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

        if not text:
            continue

        documents.extend(_chunk_text_with_metadata(file_path, text))
        file_count += 1

    if not documents:
        return None, 0

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, file_count


@tool(args_schema=FetchSummarizeAndSaveInput)
def fetch_summarize_and_save(url: str) -> str:
    """Fetch a webpage, extract key text, and save short notes as a .txt file in /docs."""
    try:
        docs_path = _ensure_docs_path()

        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LangChainNotesBot/1.0)"},
        )
        response.raise_for_status()

        notes_text = _extract_webpage_notes(response.text, url)
        if not notes_text:
            return "Error: No useful content found on the webpage."

        soup = BeautifulSoup(response.text, "html.parser")
        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else urlparse(url).netloc
        )
        safe_name = _safe_filename(title)

        file_path = docs_path / f"{safe_name}.txt"
        file_path.write_text(notes_text, encoding="utf-8")

        _invalidate_doc_index()

        return f"Saved summarized notes to {file_path}"

    except requests.RequestException as exc:
        return f"Error fetching webpage: {exc}"
    except Exception as exc:
        return f"Error processing webpage: {exc}"


@tool(args_schema=CalculateInput)
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def get_current_time() -> str:
    """Get the current date and time in UTC."""
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"


@tool(args_schema=SearchDocumentsInput)
def search_documents(query: str) -> str:
    """Search the saved docs collection and return relevant excerpts with metadata."""
    try:
        vectorstore = _get_docs_vectorstore()

        if vectorstore is None:
            return "No indexed documents found. Add .txt files to the docs directory first."

        results = vectorstore.similarity_search(query, k=MAX_SEARCH_RESULTS)

        if not results:
            return "No relevant documents found."

        formatted_results = []
        for doc in results:
            metadata = doc.metadata
            header = (
                f"File: {metadata.get('file_name', 'unknown')} | "
                f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}"
            )
            formatted_results.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_results)

    except Exception as exc:
        return f"Error searching documents: {exc}"


@tool(args_schema=ListFilesInput)
def list_files(root_dir: str | None = None) -> str:
    """List supported text/code files under a project directory."""
    try:
        root = _resolve_project_root(root_dir)

        if not root.exists():
            return f"Error: Directory does not exist: {root}"

        if not root.is_dir():
            return f"Error: Not a directory: {root}"

        files = []
        for file_path in _iter_project_files(root):
            try:
                relative_path = file_path.relative_to(root)
            except ValueError:
                relative_path = file_path
            files.append(str(relative_path))

        if not files:
            return f"No supported files found under {root}"

        files = sorted(files)

        if len(files) > MAX_FILE_LIST:
            shown = files[:MAX_FILE_LIST]
            remaining = len(files) - MAX_FILE_LIST
            return (
                f"Project root: {root}\n"
                f"Showing first {MAX_FILE_LIST} files:\n"
                + "\n".join(shown)
                + f"\n... and {remaining} more files."
            )

        return f"Project root: {root}\n" + "\n".join(files)

    except Exception as exc:
        return f"Error listing files: {exc}"


@tool(args_schema=IndexProjectInput)
def index_project(root_dir: str | None = None) -> str:
    """Index project files into a vectorstore for semantic code search."""
    try:
        root = _resolve_project_root(root_dir)

        if not root.exists():
            return f"Error: Directory does not exist: {root}"

        if not root.is_dir():
            return f"Error: Not a directory: {root}"

        vectorstore, file_count = _build_project_vectorstore(root)

        if vectorstore is None:
            return f"No indexable files found under {root}"

        global _project_vectorstore, _indexed_project_root, _indexed_project_file_count
        _project_vectorstore = vectorstore
        _indexed_project_root = root
        _indexed_project_file_count = file_count

        return f"Indexed {file_count} files from {root}"

    except Exception as exc:
        return f"Error indexing project: {exc}"


@tool(args_schema=SearchCodebaseInput)
def search_codebase(query: str) -> str:
    """Search the indexed codebase semantically and return relevant chunks with metadata."""
    try:
        if _project_vectorstore is None:
            return "No project index found. Run index_project(root_dir) first."

        results = _project_vectorstore.similarity_search(query, k=MAX_SEARCH_RESULTS)

        if not results:
            return "No relevant code found."

        formatted_results = []
        for doc in results:
            metadata = doc.metadata
            header = (
                f"File: {metadata.get('source', 'unknown')} | "
                f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')} | "
                f"Chunk: {metadata.get('chunk_index', '?')}"
            )
            formatted_results.append(f"{header}\n{doc.page_content}")

        root_text = str(_indexed_project_root) if _indexed_project_root else "unknown"
        return (
            f"Indexed project root: {root_text}\n"
            f"Indexed files: {_indexed_project_file_count}\n\n"
            + "\n\n---\n\n".join(formatted_results)
        )

    except Exception as exc:
        return f"Error searching codebase: {exc}"


def get_web_search_tool():
    """Create HTTP request tools for fetching URLs."""
    toolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=True,
    )
    return toolkit.get_tools()


@tool(args_schema=ReadFileInput)
def read_file(file_path: str) -> str:
    """Read a text file from disk and return its contents."""
    try:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return f"Error: File does not exist: {path}"

        if not path.is_file():
            return f"Error: Not a file: {path}"

        return path.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        return "Error: File is not valid UTF-8 text."
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str) -> str:
    """Write full content to a file, replacing existing content or creating the file."""
    try:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        _invalidate_project_index()
        _invalidate_doc_index()

        return f"Wrote file: {path}"

    except Exception as exc:
        return f"Error writing file: {exc}"


@tool(args_schema=ReplaceTextInput)
def replace_in_file(file_path: str, old_text: str, new_text: str) -> str:
    """Replace exact text inside a file."""
    try:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return f"Error: File does not exist: {path}"

        if not path.is_file():
            return f"Error: Not a file: {path}"

        text = path.read_text(encoding="utf-8")

        if old_text not in text:
            return "Error: old_text was not found in the file."

        updated_text = text.replace(old_text, new_text, 1)
        path.write_text(updated_text, encoding="utf-8")

        _invalidate_project_index()
        _invalidate_doc_index()

        return f"Updated file: {path}"

    except UnicodeDecodeError:
        return "Error: File is not valid UTF-8 text."
    except Exception as exc:
        return f"Error replacing text in file: {exc}"
