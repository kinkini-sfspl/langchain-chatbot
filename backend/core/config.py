from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PG_DSN: str
    class Config:
        env_file = ".env"

settings = Settings()