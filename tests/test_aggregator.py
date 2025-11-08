from evaluation import EvaluationEngine, RougeMetric


def test_compute_and_matrix_basic():
    rows = [
        {"method": "nbce", "sampling_rate": 0.2, "window_size": 256, "ref": "患者 入院", "output": "患者 入院"},
        {"method": "nbce", "sampling_rate": 0.2, "window_size": 256, "ref": "患者 治療", "output": "患者 治療"},
        {"method": "pcw", "sampling_rate": 0.2, "window_size": 512, "ref": "患者 入院", "output": "入院"},
    ]
    engine = EvaluationEngine(metrics=[RougeMetric(["rouge1"])])
    enriched = engine.score_rows(rows)
    assert all("rouge1_f" in r for r in enriched)
    matrix = engine.aggregate(
        enriched,
        group_keys=["method", "sampling_rate", "window_size"],
        metric_keys=["rouge1_f"],
    )
    # Expect two groups
    assert len(matrix) == 2
    # Check count aggregated
    nbce_row = next(m for m in matrix if m["method"] == "nbce")
    assert nbce_row["count"] == 2
