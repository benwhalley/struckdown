"""Evidence search action: BM25 keyword search over text the caller loaded.

The action searches ``context["_evidence_store"]``, a list of
``{"text", "filename"}`` chunks. It never reads files itself, so a template
cannot point it at one: whoever calls struckdown loads the evidence, with
:func:`load_evidence` for a folder of .txt/.md files, or (in the playground)
from uploads.

    from struckdown.actions.evidence import load_evidence
    complete(template, context={"_evidence_store": load_evidence(["evidence/"])})

Template:
    [[@evidence:guidance|query="CBT techniques",n=3]]
"""

from pathlib import Path

from rank_bm25 import BM25Okapi

from . import Actions


def search_remote_evidence(
    chunks: list[dict], query: str | list[str], n: int = 3
) -> str:
    """Search pre-chunked evidence from remote store.

    Args:
        chunks: List of chunk dicts with 'text' and 'filename' keys
        query: Search query text or list of queries
        n: Number of results to return

    Returns:
        Concatenated matching chunks separated by newlines
    """
    if not chunks:
        return ""

    corpus = [c["text"] for c in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    # Normalise query to list
    queries = [query] if isinstance(query, str) else list(query)

    # Collect results from all queries, dedupe by chunk index
    seen_indices = {}  # idx -> best score

    for q in queries:
        tokenized_query = q.lower().split()
        scores = bm25.get_scores(tokenized_query)

        for idx, score in enumerate(scores):
            if score > 0:
                if idx not in seen_indices or score > seen_indices[idx]:
                    seen_indices[idx] = score

    # Sort by score descending, take top n
    top_indices = sorted(
        seen_indices.keys(), key=lambda i: seen_indices[i], reverse=True
    )[:n]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        filename = chunk.get("filename", "unknown")
        text = chunk["text"]
        results.append(f"[{filename}]\n{text}")

    return "\n\n---\n\n".join(results)


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of approximately chunk_size characters.

    Splits on word boundaries to avoid cutting words in half.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def load_evidence(folders) -> list[dict]:
    """Every .txt and .md file under ``folders`` (paths or strings), chunked
    for ``context["_evidence_store"]``."""
    chunks = []
    seen = set()
    for folder in (Path(f).resolve() for f in folders):
        if folder in seen or not folder.exists():
            continue
        seen.add(folder)
        for filepath in sorted(folder.glob("**/*")):
            if filepath.is_file() and filepath.suffix in (".txt", ".md"):
                chunks.extend(
                    {"text": chunk, "filename": filepath.name}
                    for chunk in chunk_text(filepath.read_text())
                )
    return chunks


@Actions.register("evidence", on_error="return_empty", default_save=True)
def evidence_search(context: dict, query: str | list[str], n: int = 3) -> str:
    """Search ``context["_evidence_store"]`` with BM25.

    Args:
        context: Struckdown context; its '_evidence_store' is what is searched
        query: Search query text or list of queries
        n: Number of results per query

    Returns:
        Concatenated matching chunks separated by newlines ("" if no evidence)
    """
    return search_remote_evidence(context.get("_evidence_store") or [], query, n)
