import asyncio
import json
import logging
import os
import time

import openai

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ.get("LLAMACPP_API_KEY", "")

# Le serveur LLM partagé recharge parfois son modèle (503 "Loading model") quand
# un autre service en sollicite un différent — quelques tentatives espacées
# absorbent cette instabilité transitoire sans intervention manuelle.
RETRY_ATTEMPTS = int(os.environ.get("LLM_RETRY_ATTEMPTS", "4"))
RETRY_BACKOFF_SECONDS = float(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", "4"))

# Hérité de l'époque llama.cpp (lancé avec --parallel 1, donc décodage forcément
# sérialisé côté serveur — une concurrence client plus élevée n'aurait rien apporté).
# Le backend est maintenant vLLM avec continuous batching (vérifié empiriquement :
# 4 requêtes concurrentes traitées en ~0.9s au total plutôt qu'en file), donc limiter
# le client à 1 appel à la fois ne fait plus que ralentir inutilement les traitements
# par lots (classification, etc.) sans bénéfice de stabilité.
_LLM_SEMAPHORE = asyncio.Semaphore(8)
_client = None


def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL, max_retries=5)
    return _client


def _call(system: str, user: str, tool: list, max_tokens: int | None = None) -> dict:
    kwargs: dict = dict(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=tool,
        tool_choice="required",
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    resp = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = get_client().chat.completions.create(**kwargs, timeout=120)
            break
        except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
            if attempt == RETRY_ATTEMPTS:
                raise
            logger.warning(
                f"LLM call échoué (tentative {attempt}/{RETRY_ATTEMPTS}): {e} — "
                f"nouvel essai dans {RETRY_BACKOFF_SECONDS}s"
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    calls = resp.choices[0].message.tool_calls
    if not calls:
        return {}

    raw = calls[0].function.arguments
    if resp.choices[0].finish_reason == "length":
        logger.warning(f"LLM output truncated (finish=length), {len(raw)} chars")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON parse error: {e} — raw length={len(raw)}")
        return {}


async def call_tool(system: str, user: str, tool: list, max_tokens: int | None = None) -> dict:
    loop = asyncio.get_event_loop()
    async with _LLM_SEMAPHORE:
        try:
            return await loop.run_in_executor(None, _call, system, user, tool, max_tokens)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {}
