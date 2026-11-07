import pytest

from app.eval.provenance import claim_status, require_primary
from app.config import Settings


def test_same_family_llm_judge_is_blocked():
    status, reason = claim_status("ours_silver", "claude")
    assert status == "blocked"
    assert "circularity" in reason


def test_partial_human_audit_is_not_promoted_to_full_ranking_gold():
    assert claim_status("human_internal", "claude")[0] == "diagnostic"


def test_djinni_human_gold_is_primary():
    assert claim_status("djinni_human", "claude")[0] == "primary"
    require_primary("djinni_human", "claude")


def test_non_primary_evidence_fails_closed():
    with pytest.raises(ValueError, match="not primary evidence"):
        require_primary("ours_silver", "claude")


def test_llm_shortlist_ranking_is_locked_by_default():
    assert Settings(_env_file=None).max_llm_ranks == 0
