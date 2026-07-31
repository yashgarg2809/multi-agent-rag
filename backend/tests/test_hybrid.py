"""Pure-function tests: RRF fusion + tokenization (no DB, no LLMs)."""
from app.hybrid import rrf_fuse, tokenize


def test_tokenize_basic():
    assert tokenize("Stumps & BAILS!") == ["stumps", "bails"]
    assert tokenize("") == []


def test_rrf_prefers_consensus():
    # 'a' ranked #1 by both systems beats 'b' ranked #1 by one system only
    fused = dict(rrf_fuse([["a", "c"], ["b", "a"]]))
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_rrf_single_ranking_keeps_order():
    assert [k for k, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_formula():
    # rank 0 in one ranking with K=60 -> 1/61
    fused = dict(rrf_fuse([["only"]], k=60))
    assert abs(fused["only"] - 1 / 61) < 1e-9
