import pytest
from pydantic import ValidationError

from src.app.interfaces.mcp_agent import jsonschema_to_args_schema


def test_anyof_integer_schema_allows_numbers() -> None:
    schema = {
        "properties": {
            "limit": {
                "description": "Optional limit",
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "null"},
                ],
            }
        },
        "required": [],
    }

    TransactionsArgs = jsonschema_to_args_schema("TransactionsArgs", schema)

    assert TransactionsArgs(limit=15).limit == 15
    assert TransactionsArgs(limit="20").limit == "20"
    assert TransactionsArgs().limit is None


def test_place_order_like_schema_accepts_numeric_values() -> None:
    schema = {
        "properties": {
            "quantity": {
                "description": "Order quantity",
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"type": "number"},
                ],
            },
            "limit_price": {
                "description": "Limit price",
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "null"},
                ],
            },
        },
        "required": ["quantity"],
    }

    PlaceOrderArgs = jsonschema_to_args_schema("PlaceOrderArgs", schema)

    payload = PlaceOrderArgs(quantity=25, limit_price=210.0)
    assert payload.quantity == 25
    assert payload.limit_price == 210.0

    with pytest.raises(ValidationError):
        PlaceOrderArgs()


def test_string_schema_with_price_keyword_allows_numbers() -> None:
    schema = {
        "properties": {
            "limit_price": {
                "type": "string",
                "description": "Limit price",
            }
        },
        "required": ["limit_price"],
    }

    Args = jsonschema_to_args_schema("FallbackPriceArgs", schema)

    payload = Args(limit_price=210.0)
    assert payload.limit_price == 210.0

    payload2 = Args(limit_price=25)
    assert payload2.limit_price == 25
