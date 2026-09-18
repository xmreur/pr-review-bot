import httpx
import logging

LLM_URL = 'http://127.0.0.1:8080/v1/chat/completions'

# Non-streaming completions send nothing until inference finishes,
# so this must cover queue + prompt processing + generation on a
# slow local server, not just network latency.
LLM_TIMEOUT = 600
LLM_MAX_RETRIES = 3

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Shared client: reuses the localhost keep-alive connection
    instead of paying TCP+pool setup per LLM call."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=LLM_TIMEOUT)
    return _client


async def ask_llm(
    system: str,
    user: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> str:
    payload = {
        'model': "qwen",
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_object"
        }
    }

    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = await client.post(LLM_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            logger.warning(
                "LLM call timed out (attempt %d/%d): %s",
                attempt,
                LLM_MAX_RETRIES,
                exc,
            )
            if attempt < LLM_MAX_RETRIES:
                import asyncio

                await asyncio.sleep(2 * attempt)

    assert last_exc is not None
    raise last_exc