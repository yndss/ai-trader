"""
Клиент для работы с Finam TradeAPI
https://tradeapi.finam.ru/
"""


API_BASE_URL = "https://api.finam.ru/v1"
API_TIMEOUT = 30.0

import os
from typing import Any

import requests

class FinamAPIClient:
    """
    Клиент для взаимодействия с Finam TradeAPI

    Документация: https://tradeapi.finam.ru/
    """

    def __init__(self, access_token: str | None = None, base_url: str | None = None) -> None:
        """
        Инициализация клиента

        Args:
            access_token: Токен доступа к API (из переменной окружения FINAM_ACCESS_TOKEN)
            base_url: Базовый URL API (по умолчанию из документации)
        """
        self.access_token = access_token or os.getenv("FINAM_ACCESS_TOKEN", "")
        self.base_url = base_url or os.getenv("FINAM_API_BASE_URL", "https://api.finam.ru")
        self.session = requests.Session()

        if self.access_token:
            self.session.headers.update({
                "Authorization": f"{self.access_token}",
                "Content-Type": "application/json",
            })

    def execute_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """
        Выполнить HTTP запрос к Finam TradeAPI

        Args:
            method: HTTP метод (GET, POST, DELETE и т.д.)
            path: Путь API (например, /v1/instruments/SBER@MISX/quotes/latest)
            **kwargs: Дополнительные параметры для requests

        Returns:
            Ответ API в виде словаря

        Raises:
            requests.HTTPError: Если запрос завершился с ошибкой
        """
        url = f"{self.base_url}{path}"

        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()

            # Если ответ пустой (например, для DELETE)
            if not response.content:
                return {"status": "success", "message": "Operation completed"}

            return response.json()

        except requests.exceptions.HTTPError as e:
            # Пытаемся извлечь детали ошибки из ответа
            error_detail = {"error": str(e), "status_code": e.response.status_code if e.response else None}

            try:
                if e.response and e.response.content:
                    error_detail["details"] = e.response.json()
            except Exception:
                error_detail["details"] = e.response.text if e.response else None

            return error_detail

        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Получить текущую котировку инструмента"""
        return self.execute_request("GET", f"/v1/instruments/{symbol}/quotes/latest")
    def get_session_details(self) -> dict[str, Any]:
        """Получить детали текущей сессии"""
        return self.execute_request("GET", "/v1/sessions/details")


__all__ = ["FinamAPIClient"]
