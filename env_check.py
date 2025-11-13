from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV = str(Path(__file__).resolve().parent / ".env")

class S(BaseSettings):
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_TEMPERATURE: float = 0.2          # ← added
    PG_DSN: str
    model_config = SettingsConfigDict(env_file=ROOT_ENV)

s = S()
print("Loaded .env from:", ROOT_ENV)
print("GOOGLE_API_KEY length:", len(s.GOOGLE_API_KEY))
print("GEMINI_MODEL:", s.GEMINI_MODEL)
print("GEMINI_TEMPERATURE:", s.GEMINI_TEMPERATURE)
print("PG_DSN startswith 'postgresql':", s.PG_DSN.startswith("postgresql"))

# ---- DB connectivity smoke test ----
try:
    import psycopg2
    # psycopg2 expects 'postgresql', not 'postgresql+psycopg2'
    dsn = s.PG_DSN.replace("postgresql+psycopg2", "postgresql")
    try:
        import psycopg2 as _pg      # keep if psycopg2-binary works
    except ImportError:
        import psycopg as _pg       # fallback to psycopg v3

# then use:
    conn = _pg.connect(s.PG_DSN.replace("postgresql+psycopg2","postgresql"))

    #conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute("SELECT 1")
    print("DB OK:", cur.fetchone())
    cur.close(); conn.close()
except Exception as e:
    print("DB check failed:", e)

# ---- Gemini ping ----
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=s.GEMINI_MODEL,
        api_key=s.GOOGLE_API_KEY,
        temperature=s.GEMINI_TEMPERATURE,
    )
    resp = llm.invoke("Say 'ok' if you can read this.")
    print("Gemini OK:", getattr(resp, "content", str(resp))[:60])
except Exception as e:
    print("Gemini check failed:", e)
