from pathlib import Path
from tempfile import TemporaryDirectory

from forgeai.ingestion.chunker import chunk_document
from forgeai.ingestion.loader import load_document, load_folder
from forgeai.ingestion.vector_store import LocalTextStore


def test_pdf_txt_and_csv_loaders(tmp_path: Path) -> None:
    (tmp_path / "policy.txt").write_text("Annual leave is 20 days.", encoding="utf-8")
    (tmp_path / "catalog.csv").write_text("sku,name\nA1,Bearing\n", encoding="utf-8")

    documents = load_folder(tmp_path)
    assert {document.document_type for document in documents} == {"text", "csv"}


def test_chunking_and_retrieval() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "policy.md"
        path.write_text("Shore staff receive 20 days annual leave.", encoding="utf-8")
        document = load_document(path)
        chunks = chunk_document(document, chunk_size=40, chunk_overlap=10)
        store = LocalTextStore(chunks)

        results = store.search("how much annual leave?", top_k=3)
        assert results
        assert results[0].chunk.source == "policy.md"
