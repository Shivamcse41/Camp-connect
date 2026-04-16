"""
smart_rag.py
------------
3-Phase Smart Query Pipeline for CampConnect Navigator.

Phase 1: Search local FAISS vector store.
Phase 2: If confidence is low, crawl gpaurangabad.ac.in for live data.
Phase 3: LLM merges both sources into a final, validated answer.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
RELEVANCE_THRESHOLD = 0.55   # FAISS similarity score below this → fallback to web
TOP_K = 3                    # Number of FAISS chunks to retrieve

# Devanagari Unicode range — used for Hindi detection
_HINDI_RE = re.compile(r"[\u0900-\u097F]")

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embeddings = None
_vectorstore = None
_llm = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_community.vectorstores import FAISS
        _vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq
        _llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            temperature=0.4,
            max_tokens=700,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


# ── Language detection ────────────────────────────────────────────────────────
def is_hindi(text: str) -> bool:
    """Return True if the text contains Devanagari characters (Hindi)."""
    return bool(_HINDI_RE.search(text))


# ── Phase 1: Local FAISS search ───────────────────────────────────────────────
def _phase1_local_search(query: str) -> tuple[str, float]:
    """
    Search the local FAISS DB.
    Returns (combined_text, best_relevance_score).
    Score is between 0 (bad) and 1 (perfect).
    """
    try:
        vs = _get_vectorstore()
        # similarity_search_with_relevance_scores returns (Document, score) pairs
        results = vs.similarity_search_with_relevance_scores(query, k=TOP_K)
        if not results:
            return "", 0.0

        best_score = max(score for _, score in results)
        combined_text = "\n\n".join(doc.page_content for doc, _ in results)
        print(f"[Smart RAG] Phase 1 — best relevance score: {best_score:.2f}")
        return combined_text, best_score
    except Exception as e:
        print(f"[Smart RAG] Phase 1 error: {e}")
        return "", 0.0


# ── Phase 2: Web crawler ──────────────────────────────────────────────────────
def _phase2_web_crawl(query: str) -> dict:
    """Delegate to college_crawler and return its result dict."""
    from college_crawler import fetch_context_for_query
    result = fetch_context_for_query(query)
    print(f"[Smart RAG] Phase 2 — crawled: {result['url']} | status: {result['status']}")
    return result


# ── Phase 3: LLM answer generation ───────────────────────────────────────────
SYSTEM_PROMPT = """### ROLE:
You are the "CAMPCONNECT Smart Navigator." Your primary job is to provide accurate, helpful information about Government Polytechnic Aurangabad (GP Aurangabad), Bihar.

### INSTRUCTIONS:
- Answer ONLY using the context provided below. Do NOT make up facts.
- If the context does not contain the answer, say so clearly.
- Format your answer in a clear, bulleted list.
- If the user wrote in Hindi (Devanagari script), reply in Hindi. Otherwise reply in English.
- Be concise but complete.

### CONSTRAINTS:
- Do not hallucinate URLs. Only mention URLs if they appear in the context.
- If the website was down, tell the user clearly.
"""

def _phase3_generate_answer(
    query: str,
    local_context: str,
    web_result: dict | None,
    use_hindi: bool,
) -> tuple[str, str]:
    """
    Ask the LLM to combine local + web context into a final answer.
    Returns (answer_text, source_label).
    """
    llm = _get_llm()

    # Build context block
    context_parts = []
    source_label = "🗄️ Local Knowledge Base"

    if local_context:
        context_parts.append(f"[LOCAL KNOWLEDGE BASE]\n{local_context}")

    if web_result:
        if web_result["status"] == "ok":
            context_parts.append(
                f"[OFFICIAL WEBSITE - {web_result['url']}]\n{web_result['content']}"
            )
            source_label = "🌐 Official Website"
            if local_context:
                source_label = "🗄️ Local DB + 🌐 Official Website"
        elif web_result["status"] == "down":
            context_parts.append(
                "[OFFICIAL WEBSITE] ⚠️ The college website is currently unreachable."
            )

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

    lang_instruction = (
        "Please respond in Hindi (Devanagari script)."
        if use_hindi
        else "Please respond in English."
    )

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"LANGUAGE: {lang_instruction}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"ANSWER:"
    )

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=full_prompt)])
        answer = response.content.strip()
    except Exception as e:
        answer = f"Sorry, I encountered an error generating the answer: {e}"

    return answer, source_label


# ── Main public function ──────────────────────────────────────────────────────
def smart_query(query: str) -> dict:
    """
    Run the full 3-phase smart RAG pipeline.

    Returns a dict:
        {
            "answer": str,
            "source": str,        # e.g. "🗄️ Local DB + 🌐 Official Website"
            "web_url": str | None # URL crawled (if any)
        }
    """
    use_hindi = is_hindi(query)
    web_result = None

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    local_context, score = _phase1_local_search(query)
    confident = score >= RELEVANCE_THRESHOLD

    # ── Phase 2 (only if Phase 1 was not confident) ───────────────────────────
    if not confident:
        print(f"[Smart RAG] Phase 1 score {score:.2f} < threshold {RELEVANCE_THRESHOLD} -> triggering web crawl")
        web_result = _phase2_web_crawl(query)
    else:
        print(f"[Smart RAG] Phase 1 confident -> skipping web crawl")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    answer, source = _phase3_generate_answer(query, local_context, web_result, use_hindi)

    return {
        "answer": answer,
        "source": source,
        "web_url": web_result["url"] if web_result else None,
    }


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "What is the admission process?",
        "placement list dikhao",
        "latest notice kya hai?",
        "मैकेनिकल इंजीनियरिंग के बारे में बताओ",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = smart_query(q)
        print(f"Source: {result['source']}")
        print(f"Answer:\n{result['answer']}")
