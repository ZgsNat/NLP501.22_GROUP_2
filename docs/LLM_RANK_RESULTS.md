# E18: historical controlled LLM-ranking diagnostic

> **WITHDRAWN FROM THE EFFECTIVENESS CLAIM.** The 0.871 → 0.955 result uses Claude as the ranking system and Claude-made silver labels. It is blocked by the evidence firewall as same-family circularity. The rubric result (0.861 → 0.859) shows no transfer. This table is retained only for reproducibility; see `docs/CIRCULARITY_AUDIT.md`.

Model `claude_code:claude-sonnet-5-5`. The top 15 of B3 ⊕ CE [hybrid] (rules + cross-encoder) are reordered by the LLM in two passes with opposite input orders, positions averaged (agent/llm_rank.py). Dev JDs only. In this historical table, `ours` means **`ours-silver`**, never human Gold; `coding` means **`coding-rubric`**.

| bench | JD | B3 ⊕ CE | + LLM rank | P6 | S1 | traps@10 B3 → LLM | pass agreement@5 | tokens |
|---|---|---|---|---|---|---|---|---|
| ours-silver (blocked) | JD-001 | 0.799 | 0.955 | 0.799 | 0.953 | 0 → 0 | 100% | 35,464 |
| ours-silver (blocked) | JD-002 | 0.980 | 0.990 | 0.980 | 0.990 | 0 → 0 | 100% | 34,307 |
| ours-silver (blocked) | JD-003 | 0.862 | 0.926 | 0.836 | 0.832 | 0 → 0 | 80% | 31,737 |
| ours-silver (blocked) | JD-006 | 0.857 | 0.877 | 0.857 | 0.934 | 0 → 0 | 100% | 33,375 |
| ours-silver (blocked) | JD-008 | 0.949 | 0.984 | 0.949 | 0.945 | 0 → 0 | 100% | 36,372 |
| ours-silver (blocked) | JD-012 | 0.781 | 1.000 | 0.781 | 1.000 | 0 → 0 | 100% | 32,476 |
| coding-rubric (diagnostic) | ambiguous_junior | 0.884 | 0.786 | 0.886 | — | 0 → 0 | 80% | 21,254 |
| coding-rubric (diagnostic) | mobile_offshore | 0.837 | 0.932 | 0.837 | — | 0 → 0 | 100% | 22,378 |
| **ours-silver mean (blocked)** | | **0.871** | **0.955** | **0.867** | **0.942** | | | |
| **coding-rubric mean (diagnostic)** | | **0.861** | **0.859** | **0.862** | — | | | |
