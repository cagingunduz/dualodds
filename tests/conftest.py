import pytest

from dualodds.data.synthetic import generate_league
from dualodds.models import DixonColesModel


@pytest.fixture(scope="session")
def league():
    return generate_league(n_teams=12, seasons=3, seed=11)


@pytest.fixture(scope="session")
def fitted(league):
    df, _ = league
    return DixonColesModel().fit(df)
