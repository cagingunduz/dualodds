import pandas as pd

from dualodds.data.sources import FootballDataUK, registered_sources
from dualodds.data.sources.football_data_uk import parse_csv


def test_parse_csv_prefers_pinnacle_and_closing():
    raw = pd.DataFrame(
        {
            "Div": ["T1", "T1"],
            "Date": ["09/08/2024", "10/08/24"],
            "HomeTeam": ["Galatasaray", "Besiktas"],
            "AwayTeam": ["Hatayspor", "Antalyaspor"],
            "FTHG": [2, 4],
            "FTAG": [1, 2],
            "B365H": [1.2, 1.4],
            "B365D": [7.0, 5.0],
            "B365A": [13.0, 7.5],
            "PSH": [1.22, 1.42],
            "PSD": [7.2, 5.1],
            "PSA": [13.5, 7.8],
            "PSCH": [1.2, 1.4],
            "PSCD": [7.5, 5.2],
            "PSCA": [14.0, 8.0],
        }
    )
    df = parse_csv(raw, league="T1", season="2425")
    assert list(df["home"]) == ["Galatasaray", "Besiktas"]
    assert df["odds_home"].iloc[0] == 1.22
    assert df["close_away"].iloc[1] == 8.0
    assert df["date"].iloc[1] == pd.Timestamp("2024-08-10")


def test_source_registered_with_license_metadata():
    infos = {i.name: i for i in registered_sources()}
    assert FootballDataUK.info.name in infos
    assert infos[FootballDataUK.info.name].license
