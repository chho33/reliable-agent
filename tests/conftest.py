import pytest

from backends.mock import MockBackend


@pytest.fixture
def backend() -> MockBackend:
    return MockBackend()
