from pydantic import BaseModel, Field


class FetchSummarizeAndSaveInput(BaseModel):
    url: str = Field(
        description="The webpage URL to fetch, extract text from, and save as notes."
    )


class CalculateInput(BaseModel):
    expression: str = Field(
        description="A mathematical expression to evaluate, such as '(5 + 3) * 2' or 'sqrt(16)'."
    )


class SearchDocumentsInput(BaseModel):
    query: str = Field(
        description="Natural language query used to search the saved docs collection."
    )


class ListFilesInput(BaseModel):
    root_dir: str | None = Field(
        default=None,
        description="Project root directory to scan for files. If omitted, the default project root is used.",
    )


class IndexProjectInput(BaseModel):
    root_dir: str | None = Field(
        default=None,
        description="Project root directory to index for semantic code search. If omitted, the default project root is used.",
    )


class SearchCodebaseInput(BaseModel):
    query: str = Field(
        description="Natural language query describing the code, logic, function, or behavior to search for in the indexed codebase."
    )


class ReadFileInput(BaseModel):
    file_path: str = Field(
        description="Absolute or project-relative path to a text/code file to read."
    )


class WriteFileInput(BaseModel):
    file_path: str = Field(
        description="Absolute or project-relative path to the file to write."
    )
    content: str = Field(
        description="Full file content to write. This replaces existing content or creates a new file."
    )


class ReplaceTextInput(BaseModel):
    file_path: str = Field(
        description="Absolute or project-relative path to the file to update."
    )
    old_text: str = Field(description="Exact text to find in the file.")
    new_text: str = Field(description="Replacement text to insert instead of old_text.")
