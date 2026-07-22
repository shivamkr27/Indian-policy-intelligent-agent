"""Unit tests for core/ingestion.py — _ParentStore SQLite backend, text extraction, chunking."""

import json
import pytest
from core.ingestion import _ParentStore, _extract_text, _DocumentChunker


class TestParentStoreSQLite:
    @pytest.fixture
    def store(self, tmp_path):
        return _ParentStore(db_path=str(tmp_path / "parents.db"))

    def test_save_and_load_roundtrip(self, store):
        store.save("doc_parent_0", "Full parent text here.", {"source": "doc.pdf"})
        result = store.load("doc_parent_0")
        assert result["content"] == "Full parent text here."
        assert result["metadata"]["source"] == "doc.pdf"

    def test_load_missing_returns_empty(self, store):
        assert store.load("nonexistent_id") == {}

    def test_all_sources_returns_unique_filenames(self, store):
        store.save("doc1_parent_0", "text", {"source": "rbi_report.pdf"})
        store.save("doc1_parent_1", "text", {"source": "rbi_report.pdf"})
        store.save("doc2_parent_0", "text", {"source": "budget.pdf"})

        sources = store.all_sources()
        assert set(sources) == {"rbi_report.pdf", "budget.pdf"}

    def test_save_overwrites_existing(self, store):
        store.save("doc_parent_0", "original", {"source": "a.pdf"})
        store.save("doc_parent_0", "updated", {"source": "a.pdf"})
        result = store.load("doc_parent_0")
        assert result["content"] == "updated"

    def test_clear_removes_all_rows(self, store):
        store.save("p1", "text", {"source": "a.pdf"})
        store.save("p2", "text", {"source": "b.pdf"})
        store.clear()
        assert store.all_sources() == []
        assert store.load("p1") == {}

    def test_metadata_is_preserved_as_dict(self, store):
        meta = {"source": "doc.pdf", "page": 3, "heading": "Section 2"}
        store.save("doc_parent_0", "content", meta)
        result = store.load("doc_parent_0")
        assert result["metadata"] == meta

    def test_all_sources_empty_on_fresh_store(self, store):
        assert store.all_sources() == []

    def test_user_id_isolation_all_sources(self, store):
        store.save("a_parent_0", "text", {"source": "alice.pdf"}, user_id="alice")
        store.save("b_parent_0", "text", {"source": "bob.pdf"},   user_id="bob")
        assert store.all_sources(user_id="alice") == ["alice.pdf"]
        assert store.all_sources(user_id="bob")   == ["bob.pdf"]

    def test_default_user_sees_all_sources(self, store):
        store.save("a_parent_0", "text", {"source": "doc1.pdf"}, user_id="default")
        store.save("b_parent_0", "text", {"source": "doc2.pdf"}, user_id="default")
        # "default" user_id → no filter applied
        assert set(store.all_sources(user_id="default")) == {"doc1.pdf", "doc2.pdf"}

    def test_save_with_user_id_roundtrip(self, store):
        store.save("p1", "content", {"source": "doc.pdf"}, user_id="carol")
        result = store.load("p1")
        assert result["content"] == "content"

    def test_clear_per_user(self, store):
        store.save("a_p0", "text", {"source": "a.pdf"}, user_id="alice")
        store.save("b_p0", "text", {"source": "b.pdf"}, user_id="bob")
        store.clear(user_id="alice")
        assert store.all_sources(user_id="alice") == []
        assert store.all_sources(user_id="bob")   == ["b.pdf"]


class TestExtractText:
    def test_txt_file_read_directly(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Plain text content.\nSecond line.", encoding="utf-8")
        assert _extract_text(f) == "Plain text content.\nSecond line."

    def test_docx_file_extracts_paragraphs(self, tmp_path):
        docx = pytest.importorskip("docx")
        f = tmp_path / "report.docx"
        doc = docx.Document()
        doc.add_paragraph("First paragraph.")
        doc.add_paragraph("")  # blank paragraphs should be dropped
        doc.add_paragraph("Second paragraph.")
        doc.save(str(f))

        text = _extract_text(f)
        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        with pytest.raises(ValueError, match="Unsupported file type"):
            _extract_text(f)


class TestDocumentChunkerSourceMetadata:
    def test_source_metadata_matches_actual_filename_not_hardcoded_pdf(self):
        chunker = _DocumentChunker()
        long_text = "This is plain text with no markdown headers. " * 200

        parent_pairs, child_chunks = chunker.chunk(long_text, "notes.txt")

        assert parent_pairs, "expected at least one parent chunk"
        for _, parent_doc in parent_pairs:
            assert parent_doc.metadata["source"] == "notes.txt"
        for child in child_chunks:
            assert child.metadata["source"] == "notes.txt"

    def test_source_metadata_preserves_pdf_filename(self):
        chunker = _DocumentChunker()
        text = "# Heading\n\n" + ("Some markdown content. " * 200)

        parent_pairs, _ = chunker.chunk(text, "budget_2024.pdf")

        assert all(doc.metadata["source"] == "budget_2024.pdf" for _, doc in parent_pairs)
