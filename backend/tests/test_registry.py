"""Registry tests against a throwaway sqlite DB (no postgres needed)."""
import os
import tempfile

import pytest

from app import registry

url = None


@pytest.fixture(scope="module", autouse=True)
def _db():
    global url
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite:///{path}"
    registry.init_db(url)
    yield
    registry.close_engines()
    os.unlink(path)


def test_create_and_list():
    doc_id = registry.create_document("a.pdf", url)
    docs = registry.list_documents(url)
    assert any(d["id"] == doc_id and d["status"] == "indexing" for d in docs)


def test_mark_ready_and_scope_resolution():
    doc_id = registry.create_document("b.pdf", url)
    registry.mark_ready(doc_id, 42, url)
    scope = registry.ready_doc_ids(["all"], url)
    assert doc_id in scope
    scope = registry.ready_doc_ids([doc_id], url)
    assert scope == [doc_id]


def test_unknown_and_not_ready():
    with pytest.raises(KeyError):
        registry.ready_doc_ids(["missing"], url)
    pending = registry.create_document("c.pdf", url)
    with pytest.raises(ValueError):
        registry.ready_doc_ids([pending], url)


def test_mark_error_and_delete():
    doc_id = registry.create_document("d.pdf", url)
    registry.mark_error(doc_id, "boom", url)
    assert doc_id not in registry.ready_doc_ids(["all"], url)
    docs = {d["id"]: d for d in registry.list_documents(url)}
    assert docs[doc_id]["status"] == "error"
    assert registry.delete_document(doc_id, url) is True
    assert registry.delete_document(doc_id, url) is False
