import io

import pytest
from fastapi.testclient import TestClient

from app import main
from app.rag import corpus


class FakeStore:
    def __init__(self):
        self.points = {}

    def upsert_items(self, collection, items, recreate=False):
        self.points.setdefault(collection, []).extend(k for k, _, _ in items)


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(main, "upload_dir", lambda: tmp_path)
    monkeypatch.setattr(corpus, "upload_dir", lambda: tmp_path)
    corpus.load_dir.cache_clear()
    yield TestClient(main.app), store, tmp_path
    corpus.load_dir.cache_clear()


def test_upload_docx_is_parsed_redacted_and_indexed(client):
    from docx import Document

    c, store, folder = client
    doc = Document()
    for line in ["Senior QA Automation Engineer", "Email: someone@example.com", "Ngày sinh: 01/01/1990",
                 "WORK EXPERIENCE", "QA Automation Engineer — Bank (2019–2024)",
                 "• Designed a Playwright + TypeScript framework used by three teams with Page Object Model",
                 "• Wrote 400 API tests with RestAssured and ran them in Jenkins pipelines on every merge request",
                 "SKILLS", "Playwright, TypeScript, Java, RestAssured, Jenkins, Docker, familiar with Appium"]:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    r = c.post("/api/cvs", files={"file": ("cv.docx", buf.getvalue())})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["redactions"] == {"birth_date": 1, "email": 1}
    assert [s["kind"] for s in v["sections"]] == ["header", "experience", "skills"]
    skills = {s["skill"]: s["strength"] for s in v["skills"]}
    assert skills["playwright"] == "used" and skills["appium"] == "hedged"
    assert v["years_total"] == 5.0 and v["depth_cue"] >= 3
    stored = (folder / f"{v['id']}.md").read_text()
    assert "example.com" not in stored and "1990" not in stored  # only redacted text is persisted
    assert store.points["cv_chunks"] and store.points["cvs"] == [v["id"]]
    assert c.get(f"/api/cvs/{v['id']}").json()["profile"]["id"] == v["id"]


def test_upload_rejects_unsupported_or_empty(client):
    c, _, _ = client
    assert c.post("/api/cvs", files={"file": ("cv.png", b"x")}).status_code == 400
    assert c.post("/api/cvs", files={"file": ("cv.txt", b"too short")}).status_code == 400
