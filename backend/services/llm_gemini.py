
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.2

    # 👇 pydantic v2 style: load .env and ignore unrelated keys like PG_DSN
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_llm():
    s = Settings()
    return ChatGoogleGenerativeAI(
        api_key=s.GOOGLE_API_KEY,
        model=s.GEMINI_MODEL,
        temperature=s.GEMINI_TEMPERATURE,
    )
