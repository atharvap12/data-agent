import os
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load MARKETSTACK_API_KEY from the project's .env, found by path so it also works
# when the app is started from a different folder
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


#------------------------------------ API ENDPOINTS ------------------------------------#

# Pokemon list: small and mostly text (name + url). Good for simple filters.
POKEMON_API_URL = "https://pokeapi.co/api/v2/pokemon"

# Marketstack end-of-day stock prices: many numeric columns (open, high, low, close,
# volume, adjusted prices, ...) per symbol per day. Good for complex transformations
# such as aggregations, rolling averages, daily returns and pivots.
MARKETSTACK_EOD_URL = "https://api.apilayer.net/marketstack/v2/eod"


#------------------------------------ NAMED SOURCES ------------------------------------#

# Sources the ETL agent can request by name, so API keys stay out of the LLM prompt.
# `params` is a function so the key is read from the environment at call time.
API_SOURCES = {
    "pokemon": {
        "url": POKEMON_API_URL,
        "params": lambda: {},
        "description": "List of Pokemon (name, url). Mostly text, few rows.",
    },
    "marketstack": {
        "url": MARKETSTACK_EOD_URL,
        "params": lambda: {
            "access_key": os.environ.get("MARKETSTACK_API_KEY", ""),
            "symbols": "AAPL,MSFT",
        },
        "description": (
            "End-of-day stock prices for AAPL and MSFT: open, high, low, close, volume, "
            "adj_open, adj_high, adj_low, adj_close, adj_volume, split_factor, dividend, "
            "name, exchange_code, asset_type, price_currency, symbol, exchange, date."
        ),
    },
}


def resolve_source(source: str) -> tuple[str, dict]:
    """
    Turn a source into a (url, params) pair.

    Args:
        source (str): Either a registered source name (see API_SOURCES) or a full URL.

    Returns:
        tuple[str, dict]: The URL to call and the query parameters to send.
    """
    key = source.strip().lower()
    if key in API_SOURCES:
        entry = API_SOURCES[key]
        return entry["url"], entry["params"]()
    return source.strip(), {}


def source_name(source: str) -> str:
    """
    Short name of a source, used in output file names (e.g. "marketstack_20261108_101500.csv").

    - A registered name ("pokemon", "marketstack") is returned as is.
    - A URL that matches a registered source's URL returns that source's name,
      so "https://pokeapi.co/api/v2/pokemon" gives "pokemon".
    - Any other URL returns its host name, e.g. "api.example.com" gives "example".
    """
    key = source.strip().lower()
    if key in API_SOURCES:
        return key
    url = key.split("?")[0].rstrip("/")
    for name, entry in API_SOURCES.items():
        if url == entry["url"].lower().rstrip("/"):
            return name
    host = urlparse(url).hostname or "data"
    parts = [p for p in host.split(".") if p not in ("www", "api")]
    return parts[0] if parts else "data"


#------------------------------------ UNIVERSAL API CALL ------------------------------------#

def call_api(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    method: str = "GET",
    json_body: dict | None = None,
    timeout: int = 30,
):
    """
    Make an HTTP request to any JSON API and return the parsed response.

    Args:
        url (str): The API endpoint.
        params (dict, optional): Query string parameters (e.g. {"access_key": ..., "symbols": ...}).
        headers (dict, optional): Request headers (e.g. {"Authorization": "Bearer ..."}).
        method (str): HTTP method, "GET" by default.
        json_body (dict, optional): JSON body for POST/PUT/PATCH requests.
        timeout (int): Seconds to wait before giving up.

    Returns:
        dict | list: The JSON response.

    Raises:
        requests.exceptions.RequestException: If the request fails or returns an error status.
    """
    response = requests.request(
        method=method.upper(),
        url=url,
        params=params,
        headers=headers,
        json=json_body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


# Keys that APIs commonly use for the list of records in their response
RECORD_KEYS = ("results", "data", "items", "records", "rows")


def extract_records(payload) -> list:
    """
    Find the list of records inside an API response, whatever its shape.

    - PokeAPI puts them under "results"
    - Marketstack puts them under "data"
    - Some APIs return a plain list

    Args:
        payload (dict | list): The parsed JSON response.

    Returns:
        list: The records, ready for pandas.json_normalize.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in RECORD_KEYS:
            if isinstance(payload.get(key), list):
                return payload[key]
        # No known key: take the first value that is a list of objects
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        # A single object: treat it as one record
        return [payload]
    return []
