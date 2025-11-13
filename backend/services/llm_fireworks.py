# backend/services/llm_fireworks.py
from pydantic_settings import BaseSettings
from functools import lru_cache
from langchain_fireworks import ChatFireworks


class Settings(BaseSettings):
    FIREWORKS_API_KEY: str
    FIREWORKS_MODEL: str = "accounts/fireworks/models/mixtral-8x7b-instruct"
    FIREWORKS_TEMPERATURE: float = 0.2

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_llm():
    s = Settings()
    # ChatFireworks uses the Fireworks API directly (no langchain_openai needed)
    return ChatFireworks(
        api_key=s.FIREWORKS_API_KEY,
        model=s.FIREWORKS_MODEL,
        temperature=s.FIREWORKS_TEMPERATURE,
        # You can also pass max_tokens, top_p, etc., if you like
    )
