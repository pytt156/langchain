from pydantic import BaseModel, Field, model_validator


class FetchSummarizeAndSaveInput(BaseModel):
    url: str = Field(
        description="The webpage URL to fetch, extract text from, and save as notes."
    )


class FindFilesInput(BaseModel):
    filename_query: str = Field(
        description="Partial or exact filename to search for, for example 'models.py' or 'tools'."
    )
    root_dir: str | None = Field(
        default=None,
        description="Project root directory to search in. If omitted, the default project root is used.",
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
        default="",
        description="Path to the file to read. Use file_path as the argument name.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_path_alias(cls, values):
        if "path" in values and not values.get("file_path"):
            values["file_path"] = values.pop("path")
        return values


class WriteFileInput(BaseModel):
    file_path: str = Field(default="", description="Path to the file to write.")
    content: str = Field(description="Full file content to write.")

    @model_validator(mode="before")
    @classmethod
    def accept_path_alias(cls, values):
        if "path" in values and not values.get("file_path"):
            values["file_path"] = values.pop("path")
        return values
