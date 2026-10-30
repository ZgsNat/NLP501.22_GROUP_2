"""NLP501-bench v1 is frozen: the split never overlaps and no pinned input changes silently (docs/BENCHMARK.md)."""
import json

from app.config import ROOT
from app.eval import benchmark


def test_manifest_split_is_disjoint_and_covers_every_jd():
    m = json.loads(benchmark.MANIFEST.read_text())
    for src in ("ours", "coding"):
        test, dev = set(m["splits"]["test"][src]), set(m["splits"]["dev"][src])
        assert not test & dev
    assert len(m["splits"]["test"]["ours"] + m["splits"]["dev"]["ours"]) == 12
    assert benchmark.split_ours() == {k: m["splits"][k]["ours"] for k in ("test", "dev")}  # the seed reproduces it


def test_pinned_inputs_are_unchanged():
    m = json.loads(benchmark.MANIFEST.read_text())
    present = {p: h for p, h in m["sha256"].items() if (ROOT / p).exists()}  # coding files live in .runtime
    assert any(p.startswith("data/") for p in present)
    changed = [p for p, h in present.items() if benchmark._sha(ROOT / p) != h]
    assert not changed, f"benchmark inputs changed (a new version is needed): {changed[:5]}"
