import pytest

from src.app.mcp import server


@pytest.mark.asyncio
async def test_auth_updates_authorization_header(monkeypatch):
    captured: dict[str, object] = {}

    def fake_execute_request(method, path, **kwargs):  # noqa: ANN001, D401 - test double
        captured["call"] = (method, path, kwargs)
        return {"token": "jwt-token"}

    original_header = server.api_client.session.headers.get("Authorization")
    monkeypatch.setattr(server.api_client, "execute_request", fake_execute_request)

    header_after: str | None = None
    try:
        result = await server.Auth("secret-key")
        header_after = server.api_client.session.headers.get("Authorization")
    finally:
        if original_header is None:
            server.api_client.session.headers.pop("Authorization", None)
        else:
            server.api_client.session.headers["Authorization"] = original_header

    assert result == {"token": "jwt-token"}
    assert captured["call"] == (
        "POST",
        "/v1/sessions",
        {"json": {"secret": "secret-key"}},
    )
    assert header_after == "jwt-token"


@pytest.mark.asyncio
async def test_trades_builds_expected_params(monkeypatch):
    captured: dict[str, object] = {}

    def fake_execute_request(method, path, **kwargs):  # noqa: ANN001, D401 - test double
        captured["call"] = (method, path, kwargs)
        return {"trades": []}

    monkeypatch.setattr(server.api_client, "execute_request", fake_execute_request)
    server.api_client.session.headers["Authorization"] = "jwt-existing"

    result = await server.Trades(
        account_id="TRQD05:409933",
        limit="50",
        interval_start="1711929600",
        interval_end="1714521600",
    )

    assert result == {"trades": []}
    assert captured["call"] == (
        "GET",
        "/v1/accounts/TRQD05:409933/trades",
        {
            "params": {
                "limit": "50",
                "interval.start_time": "1711929600",
                "interval.end_time": "1714521600",
            }
        },
    )
    server.api_client.session.headers.pop("Authorization", None)


@pytest.mark.asyncio
async def test_bars_passes_timeframe(monkeypatch):
    captured: dict[str, object] = {}

    def fake_execute_request(method, path, **kwargs):  # noqa: ANN001, D401 - test double
        captured["call"] = (method, path, kwargs)
        return {"bars": []}

    monkeypatch.setattr(server.api_client, "execute_request", fake_execute_request)
    server.api_client.session.headers["Authorization"] = "jwt-existing"

    result = await server.Bars(
        symbol="SBER@MISX",
        timeframe="D",
        interval_start="1711929600",
        interval_end="1714521600",
    )

    assert result == {"bars": []}
    assert captured["call"] == (
        "GET",
        "/v1/instruments/SBER@MISX/bars",
        {
            "params": {
                "timeframe": "D",
                "interval.start_time": "1711929600",
                "interval.end_time": "1714521600",
            }
        },
    )
    server.api_client.session.headers.pop("Authorization", None)
