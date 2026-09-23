import pytest

from examples.scenarios import fixture_worlds


@pytest.fixture
def forest():
    return fixture_worlds()["dark_forest_village"]
