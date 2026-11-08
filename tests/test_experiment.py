from scripts.run_experiment import run


def test_experiment_writes_all_outputs(tmp_path, papers, embedder):
    questions = [
        {"question": "speech recognition for Swahili", "paper_id": "0002"},
        {"question": "German to English translation", "paper_id": "0003"},
        {"question": "classify customer reviews", "paper_id": "0004"},
    ]
    summary = run(papers, questions, embedder, out_dir=tmp_path)

    assert len(summary) == 6  # 3 retrievers x 2 chunk sizes
    assert summary["recall@5"].between(0, 1).all()
    for name in ["retrieval_results.csv", "retrieval_results.svg", "results.md"]:
        assert (tmp_path / name).exists()
    assert "Paired bootstrap tests" in (tmp_path / "results.md").read_text(encoding="utf-8")
