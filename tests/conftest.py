import pytest


@pytest.fixture(autouse=True)
def enable_local_route_tests(monkeypatch):
    import main
    monkeypatch.setattr(main, 'ACCESS_CLOSED', False)
