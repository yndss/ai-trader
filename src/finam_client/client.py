"""HTTP client for Finam TradeAPI."""

from __future__ import annotations

import os
from typing import Any

import requests


class FinamAPIClient:
    """Synchronous Finam TradeAPI client powered by ``requests``."""

    def __init__(self, access_token: str | None = None, base_url: str | None = None) -> None:
        """Initialise client with optional API token and custom base URL."""
        self.access_token = access_token or os.getenv("FINAM_ACCESS_TOKEN", "")
        self.base_url = base_url or os.getenv("FINAM_API_BASE_URL", "https://api.finam.ru")
        self.session = requests.Session()

        if self.access_token:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                }
            )

    def execute_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Perform HTTP request against Finam TradeAPI endpoint."""
        url = f"{self.base_url}{path}"

        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()

            if not response.content:
                return {"status": "success", "message": "Operation completed"}

            return response.json()

        except requests.exceptions.HTTPError as exc:
            error_detail: dict[str, Any] = {
                "error": str(exc),
                "status_code": exc.response.status_code if exc.response else None,
            }
            try:
                if exc.response and exc.response.content:
                    error_detail["details"] = exc.response.json()
            except Exception:
                error_detail["details"] = exc.response.text if exc.response else None
            return error_detail

        except Exception as exc:  # pragma: no cover - defensive catch-all
            return {"error": str(exc), "type": type(exc).__name__}

    def get_quote(self, symbol: str) -> dict[str, Any]:
        return self.execute_request("GET", f"/v1/instruments/{symbol}/quotes/latest")

    def get_orderbook(self, symbol: str, depth: int = 10) -> dict[str, Any]:
        return self.execute_request("GET", f"/v1/instruments/{symbol}/orderbook", params={"depth": depth})

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "D",
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        params = {"timeframe": timeframe}
        if start:
            params["interval.start_time"] = start
        if end:
            params["interval.end_time"] = end
        return self.execute_request("GET", f"/v1/instruments/{symbol}/bars", params=params)

    def get_account(self, account_id: str) -> dict[str, Any]:
        return self.execute_request("GET", f"/v1/accounts/{account_id}")

    def get_orders(self, account_id: str) -> dict[str, Any]:
        return self.execute_request("GET", f"/v1/accounts/{account_id}/orders")

    def get_order(self, account_id: str, order_id: str) -> dict[str, Any]:
        return self.execute_request("GET", f"/v1/accounts/{account_id}/orders/{order_id}")

    def create_order(self, account_id: str, order_data: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
        return self.execute_request("POST", f"/v1/accounts/{account_id}/orders", json=order_data)

    def cancel_order(self, account_id: str, order_id: str) -> dict[str, Any]:
        return self.execute_request("DELETE", f"/v1/accounts/{account_id}/orders/{order_id}")

    def get_trades(
        self,
        account_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if start:
            params["interval.start_time"] = start
        if end:
            params["interval.end_time"] = end
        return self.execute_request("GET", f"/v1/accounts/{account_id}/trades", params=params)

    def get_positions(self, account_id: str) -> dict[str, Any]:
        return self.execute_request("GET", f"/v1/accounts/{account_id}")

    def get_session_details(self) -> dict[str, Any]:
        return self.execute_request("POST", "/v1/sessions/details")


__all__ = ["FinamAPIClient"]
