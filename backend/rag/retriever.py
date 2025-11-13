
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

# ---- New: tighten retrieval so unrelated queries get filtered out
# Tune between 0.25–0.45 depending on your corpus size/quality
SCORE_THRESHOLD = 0.35

class Settings(BaseSettings):
    PG_DSN: str
    COLLECTION: str = "business_docs"
    EMB_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ✅ load .env and ignore unrelated keys (e.g., GOOGLE_API_KEY, GEMINI_MODEL)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def _emb():
    st = Settings()
    return HuggingFaceEmbeddings(model_name=st.EMB_MODEL)

@lru_cache
def _store():
    st = Settings()
    return PGVector(
        collection_name=st.COLLECTION,
        connection_string=st.PG_DSN,
        embedding_function=_emb(),
    )

def get_retriever(k: int = 4):
    # return _store().as_retriever(search_kwargs={"k": k})
     """
     Retriever that *drops* low-relevance hits via similarity score threshold.
     If nothing clears the bar, we treat the query as out-of-scope.
     """
     return _store().as_retriever(
         search_type="similarity_score_threshold",
         search_kwargs={"k": k, "score_threshold": SCORE_THRESHOLD},
         )
