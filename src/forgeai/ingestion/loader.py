from pathlib import Path
from typing import Iterator

import pandas as pd
from pydantic import BaseModel
from pypdf import PdfReader


class Document(BaseModel):
    source: str
    document_type: str
    text: str
    metadata: dict[str, str] = {}


def load_document(path: Path) -> Document:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
        return Document(
            source=path.name,
            document_type="pdf",
            text=text,
            metadata={"file_path": str(path), "page_count": str(len(reader.pages))},
        )
    if suffix == ".txt" or suffix in {".md", ".markdown"}:
        return Document(
            source=path.name,
            document_type="text",
            text=path.read_text(encoding="utf-8"),
            metadata={"file_path": str(path)},
        )
    if suffix == ".csv":
        frame = pd.read_csv(path)
        text = frame.to_csv(index=False)
        return Document(
            source=path.name,
            document_type="csv",
            text=text,
            metadata={
                "file_path": str(path),
                "columns": ",".join(frame.columns),
                "row_count": str(len(frame)),
            },
        )
    raise ValueError(f"Unsupported document type: {suffix}")


def load_folder(folder: Path) -> list[Document]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    documents: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".pdf", ".txt", ".md", ".markdown", ".csv"
        }:
            documents.append(load_document(path))
    return documents

