from types import SimpleNamespace

import pytest

from dualodds.agents import AnalysisContext, MatchReport
from dualodds.agents import tools
from dualodds.agents.analyst import AnalystError, analyze_match


@pytest.fixture(scope="module")
def ctx(league):
    df, _ = league
    return AnalysisContext.build(df)


def test_tools_return_serializable_data(ctx):
    probs = tools.model_probabilities(ctx, "Takim00", "Takim01")
    assert probs["home"] + probs["draw"] + probs["away"] == pytest.approx(1.0)
    form = tools.team_form(ctx, "Takim00", last_n=5)
    assert len(form["matches"]) == 5
    assert set(m["result"] for m in form["matches"]) <= {"G", "B", "M"}
    h2h = tools.head_to_head(ctx, "Takim00", "Takim01")
    assert h2h["matches"]
    assert all({m["home"], m["away"]} == {"Takim00", "Takim01"} for m in h2h["matches"])
    market = tools.market_comparison(ctx, "Takim00", "Takim01", 2.1, 3.4, 3.6)
    assert sum(o["market_fair_prob"] for o in market["outcomes"]) == pytest.approx(1.0, abs=1e-3)


class _FakeRunnerClient:
    """tool_runner'ı taklit eder: ajanın submit_report çağırdığını simüle eder."""

    def __init__(self, submit: bool = True):
        self.kwargs = None
        self.submit = submit
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._runner))

    def _runner(self, **kwargs):
        self.kwargs = kwargs
        if self.submit:
            submit = next(t for t in kwargs["tools"] if t.name == "submit_report")
            submit.call(
                {
                    "headline": "Ev sahibi hafif favori",
                    "summary": "Model ev sahibini öne çıkarıyor.",
                    "key_factors": ["İç saha avantajı"],
                    "risks": ["Rotasyon"],
                    "confidence": "orta",
                    "lean": "1",
                }
            )
        return [SimpleNamespace(stop_reason="end_turn")]


def test_analyze_match_numbers_come_from_model(ctx):
    client = _FakeRunnerClient()
    report = analyze_match(ctx, "Takim00", "Takim01", odds=(2.1, 3.4, 3.6), client=client)
    assert isinstance(report, MatchReport)
    expected = ctx.model.predict("Takim00", "Takim01")
    assert report.probabilities["home"] == pytest.approx(expected.home)
    assert len(report.value) == 3
    assert report.lean == "1"
    assert client.kwargs["model"] == "claude-opus-5-5"
    md = report.to_markdown()
    assert "Takim00 – Takim01" in md and "Değer analizi" in md


def test_analyze_match_requires_submission(ctx):
    with pytest.raises(AnalystError):
        analyze_match(ctx, "Takim00", "Takim01", client=_FakeRunnerClient(submit=False))
