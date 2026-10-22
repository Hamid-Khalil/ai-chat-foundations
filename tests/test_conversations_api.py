import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_conversation_repo, get_message_repo
from app.main import app
from tests.fakes import InMemoryConversationRepo, InMemoryMessageRepo


@pytest.fixture
def client():
    # Override dependencies with fakes
    conv_repo = InMemoryConversationRepo()
    msg_repo = InMemoryMessageRepo()
    app.dependency_overrides[get_conversation_repo] = lambda: conv_repo
    app.dependency_overrides[get_message_repo] = lambda: msg_repo

    # Create TestClient without `with` block so lifespan doesn't trigger Mongo connection
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def test_api_crud_flow(client):
    # Create
    r = client.post("/api/v1/conversations", json={"title": "Test Title"})
    assert r.status_code == 201
    cid = r.json()["id"]

    # Get
    r = client.get(f"/api/v1/conversations/{cid}")
    assert r.status_code == 200
    assert r.json()["title"] == "Test Title"

    # Patch
    r = client.patch(f"/api/v1/conversations/{cid}", json={"archived": True})
    assert r.status_code == 200

    # List (pagination cursor)
    r = client.get("/api/v1/conversations?archived=true")
    assert len(r.json()["items"]) == 1

    # Delete
    r = client.delete(f"/api/v1/conversations/{cid}")
    assert r.status_code == 200
    assert client.get(f"/api/v1/conversations/{cid}").status_code == 404


def test_api_422_and_injected_503(client):
    assert client.post("/api/v1/conversations", json={"title": ""}).status_code == 422

    # Inject DB failure
    repo = app.dependency_overrides[get_conversation_repo]()
    repo.fail_on = "create"
    r = client.post("/api/v1/conversations", json={"title": "Fail"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "DB_UNAVAILABLE"
