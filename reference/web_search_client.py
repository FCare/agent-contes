"""Client for agent-web-search's fixed service topic (not session-scoped —
any service can call it directly), mirroring agent-news's search_client.py."""

import logging
import os

from nexus_client import NexusClient

logger = logging.getLogger(__name__)

VK_URL = os.environ["VK_URL"]
MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
SERVICE_USERNAME = os.environ["MQTT_SERVICE_USERNAME"]
SERVICE_API_KEY = os.environ["MQTT_SERVICE_API_KEY"]

SERVICE_REQUEST_TOPIC = "service/search/request"
SEARCH_TIMEOUT = float(os.environ.get("WEB_SEARCH_TIMEOUT", "45"))

_nexus = None


def _get_nexus() -> NexusClient:
    global _nexus
    if _nexus is None:
        _nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)
        _nexus.start_listening()
    return _nexus


async def search(query: str, categories: str = "general", detail_level: int = 2) -> dict:
    nexus = _get_nexus()
    result = await nexus.request(
        SERVICE_REQUEST_TOPIC,
        {"query": query, "categories": categories, "n_results": 8, "detail_level": detail_level},
        timeout=SEARCH_TIMEOUT,
    )
    if result is None:
        logger.warning(f"web_search: timeout pour {query!r}")
        return {"report": "", "sources": []}
    return result
