from pydantic_settings import BaseSettings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

class S(BaseSettings):
    PG_DSN: str
    class Config:
        env_file = ".env"   # expects .env at project root

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = PGVector(
    collection_name="business_docs",
    connection_string=S().PG_DSN,
    embedding_function=emb,
)
# ret = store.as_retriever(search_kwargs={"k": 3})
# docs = ret.get_relevant_documents("test")
# print("docs:", len(docs))
# for d in docs[:2]:
#     print("-", d.page_content[:120].replace("\n"," "))

ret = store.as_retriever(search_kwargs={"k": 3})
docs = ret.invoke("test")   # v0.3 retrievers are Runnables
print("docs:", len(docs))
for d in docs[:2]:
    print("-", d.page_content[:120].replace("\n"," "))