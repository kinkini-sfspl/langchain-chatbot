
from __future__ import annotations
from typing import List
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableBranch,
)

# ---- Single source of truth for refusal text
SCOPE_MSG = "That topic is outside my scope. I can help with MFI Business Document related information."


# llm: any LangChain chat model
# retriever: any LangChain retriever that supports .invoke(question) -> List[Document]
def build_chain(llm, retriever):
    """
    Returns a Runnable that:
      input:  {"question": str, "history": list[BaseMessage]}
      output: {"answer": str}
    """
    # Prepare context from retrieved docs
    def _join_docs(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    # Build context from retrieval
    context = itemgetter("question") | retriever | RunnableLambda(_join_docs)

    # Prompt with conversation history (tightened instructions)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant for microfinance domain Q&A. "
                    "Answer strictly and briefly using ONLY the provided CONTEXT. "
                    "If the question is unrelated to microfinance documents or the CONTEXT "
                    f"is insufficient, respond exactly: \"{SCOPE_MSG}\""
                ),
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "Question: {question}\n\nCONTEXT:\n{context}",
            ),
        ]
    )

    # Inputs fan-in
    inputs = RunnableParallel(
        question=itemgetter("question"),
        history=itemgetter("history"),
        context=context,
    )

    # Branch: if context is empty -> refuse with fixed message; else go to LLM chain
    refuse_if_no_context = RunnableLambda(lambda d: {"answer": SCOPE_MSG})

    llm_chain = prompt | llm | StrOutputParser() | RunnableLambda(lambda text: {"answer": text})

    chain = inputs | RunnableBranch(
        # Condition 1: no/blank context → refuse deterministically
        (lambda d: not d.get("context", "").strip(), refuse_if_no_context),
        # Fallback: normal flow
        llm_chain,
    )

    return chain
