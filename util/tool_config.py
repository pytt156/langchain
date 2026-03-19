from pathlib import Path

DOCS_PATH = Path("/home/pytt/skola/langchain/docs")
DEFAULT_PROJECT_ROOT = Path("/home/pytt/skola/langchain")

MAX_HEADINGS = 10
MAX_PARAGRAPHS = 8
MAX_FILE_LIST = 200
MAX_SEARCH_RESULTS = 4

CHUNK_SIZE = 500
CHUNK_OVERLAP_LINES = 5

ALLOWED_TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".csv",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".sql",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
}
