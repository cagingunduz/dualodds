from dualodds.evaluation import walk_forward


def test_walk_forward_runs(league):
    df, _ = league
    start = df["date"].iloc[len(df) // 2]
    result = walk_forward(df, start=start, refit_every_days=28)
    s = result.summary
    assert s["n_matches"] > 0
    assert s["model_log_loss"] < 1.0986  # Tekdüze tahminden (ln 3) iyi
    assert "market_log_loss" in s
    assert result.predictions["date"].min() >= start
