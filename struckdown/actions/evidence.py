"""Standalone evidence search action using BM25 over text files.

Provides a simple file-based evidence search for use with `sd chat` CLI
without requiring a database or embeddings. Uses BM25 for keyword-based ranking.

Usage:
    sd chat -p prompt.sd -c evidence_folder=./my_docs

Template:
    [[@evidence:guidance|query="CBT techniques",n=3]]
"""

from pathlib import Path

from rank_bm25 import BM25Okapi

from . import Actions


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


def load_evidence_files(folders: list[Path]) -> list[tuple[str, str, str]]:
    """Load all .txt and .md files from folders.

    Returns list of (filename, chunk_text, full_path) tuples.
    """
    chunks = []
    seen_folders = set()

    for folder in folders:
        # dedupe folders
        folder = folder.resolve()
        if folder in seen_folders or not folder.exists():
            continue
        seen_folders.add(folder)

        for filepath in folder.glob("**/*"):
            if filepath.is_file() and filepath.suffix in (".txt", ".md"):
                text = filepath.read_text()
                for chunk in chunk_text(text):
                    chunks.append((filepath.name, chunk, str(filepath)))

    return chunks


@Actions.register("evidence", on_error="return_empty", default_save=True)
def evidence_search(context: dict, query: str | list[str], n: int = 3) -> str:
    """Search evidence files using BM25.

    Args:
        context: Struckdown context. Looks for:
            - evidence_folder: explicit path(s) to evidence directory
            - _template_path: auto-injected path to discover evidence/ relative to template
        query: Search query text or list of queries
        n: Number of results per query

    Returns:
        Concatenated matching chunks separated by newlines
    """
    folders = []

    # Check context for explicit folder(s)
    if evidence_folder := context.get("evidence_folder"):
        if isinstance(evidence_folder, list):
            folders.extend(Path(f) for f in evidence_folder)
        else:
            folders.append(Path(evidence_folder))

    # Auto-discover relative to template
    if template_path := context.get("_template_path"):
        evidence_dir = Path(template_path).parent / "evidence"
        if evidence_dir.exists():
            folders.append(evidence_dir)

    # Also check cwd/evidence
    cwd_evidence = Path.cwd() / "evidence"
    if cwd_evidence.exists():
        folders.append(cwd_evidence)

    if not folders:
        return ""

    # Load and index
    chunks = load_evidence_files(folders)
    if not chunks:
        return ""

    corpus = [c[1] for c in chunks]  # just the text
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
    top_indices = sorted(seen_indices.keys(), key=lambda i: seen_indices[i], reverse=True)[:n]

    results = []
    for idx in top_indices:
        filename, text, path = chunks[idx]
        results.append(f"[{filename}]\n{text}")

    return "\n\n---\n\n".join(results)
