import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import connect, init_db
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    # _env_file=None so a developer's .env never leaks into tests.
    return Settings(_env_file=None, db_path=str(tmp_path / "test.db"))


@pytest.fixture
def conn(settings):
    c = connect(settings.db_path)
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c
