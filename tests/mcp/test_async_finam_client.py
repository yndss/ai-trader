import json

import pytest

from src.app.mcp.server import AsyncFinamClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = {"content-type": "application/json"}
        self.text = json.dumps(self._payload)

    def json(self):  # noqa: D401 - emulate httpx.Response API
        return self._payload


@pytest.mark.asyncio
async def test_request_auto_auth(monkeypatch):
    client = AsyncFinamClient(base_url="https://example.com")
    client.set_jwt(None)
    client.set_secret("secret-value")

    calls: list[tuple[str, str | None]] = []

    async def fake_request(self, method, path, params=None, json=None, headers=None):  # noqa: D401
        auth_header = headers.get("Authorization") if headers else None
        calls.append((path, auth_header))

        if path == "/v1/sessions":
            return FakeResponse(200, {"jwt": "jwt-token"})

        return FakeResponse(200, {"ok": True})

    monkeypatch.setattr(client._client, "request", fake_request.__get__(client._client))

    result = await client._request("GET", "/v1/protected")

    assert result["ok"] is True
    assert calls == [("/v1/sessions", None), ("/v1/protected", "jwt-token")]


@pytest.mark.asyncio
async def test_request_refreshes_token_on_unauthorized(monkeypatch):
    client = AsyncFinamClient(base_url="https://example.com")
    client.set_jwt("stale-token")
    client.set_secret("secret-value")

    calls: list[tuple[str, str | None]] = []
    session_calls = 0

    async def fake_request(self, method, path, params=None, json=None, headers=None):  # noqa: D401
        nonlocal session_calls
        auth_header = headers.get("Authorization") if headers else None
        calls.append((path, auth_header))

        if path == "/v1/sessions":
            session_calls += 1
            return FakeResponse(200, {"jwt": f"jwt-{session_calls}"})

        # First protected call fails, forcing refresh
        if len([call for call in calls if call[0] == "/v1/protected"]) == 1:
            return FakeResponse(401, {"ok": False, "status": 401})

        return FakeResponse(200, {"ok": True})

    monkeypatch.setattr(client._client, "request", fake_request.__get__(client._client))

    result = await client._request("GET", "/v1/protected")

    assert result["ok"] is True
    assert session_calls == 1
    assert calls == [
        ("/v1/protected", "stale-token"),
        ("/v1/sessions", None),
        ("/v1/protected", "jwt-1"),
    ]
