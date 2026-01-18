"""
Text chunking utilities for evidence files.

Provides sentence-based chunking with overlap for BM25 search.
"""

import re


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split text into sentences, returning (start, end) character offsets.

    Uses a simple regex approach that handles common sentence endings.
    """
    if not text.strip():
        return []

    # Pattern: sentence-ending punctuation followed by space and capital letter
    # or end of string
    pattern = r"[.!?]+(?:\s+(?=[A-Z])|$)"

    sentences = []
    last_end = 0

    for match in re.finditer(pattern, text):
        end = match.end()
        # Strip trailing whitespace from sentence
        sentence_end = match.start() + len(match.group().rstrip())
        if sentence_end > last_end:
            sentences.append((last_end, sentence_end))
        last_end = end

    # Handle remaining text (no sentence-ending punctuation)
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append((last_end, len(text.rstrip())))

    return sentences


def chunk_text_sentences(
    text: str,
    sentences_per_chunk: int = 10,
    overlap_sentences: int = 3,
    max_chunk_chars: int = 2000,
) -> list[dict]:
    """Split text into overlapping sentence chunks.

    Args:
        text: Input text to chunk
        sentences_per_chunk: Target number of sentences per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        max_chunk_chars: Maximum characters per chunk (truncates if exceeded)

    Returns:
        List of dicts with keys: text, index, start_char, end_char
    """
    if overlap_sentences >= sentences_per_chunk:
        raise ValueError("overlap_sentences must be smaller than sentences_per_chunk")

    sentences = split_sentences(text)
    if not sentences:
        # No sentences found, return whole text as one chunk
        stripped = text.strip()
        if stripped:
            return [
                {"text": stripped, "index": 0, "start_char": 0, "end_char": len(text)}
            ]
        return []

    chunks = []
    step = sentences_per_chunk - overlap_sentences
    chunk_index = 0

    for i in range(0, len(sentences), step):
        # Get sentence range for this chunk
        end_idx = min(i + sentences_per_chunk, len(sentences))
        start_char = sentences[i][0]
        end_char = sentences[end_idx - 1][1]

        chunk_text = text[start_char:end_char].strip()

        # Truncate if too long (respecting word boundaries)
        if len(chunk_text) > max_chunk_chars:
            truncated = chunk_text[:max_chunk_chars]
            last_space = truncated.rfind(" ")
            if last_space > max_chunk_chars // 2:
                chunk_text = truncated[:last_space]

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "index": chunk_index,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )
            chunk_index += 1

    return chunks


def reconstruct_chunks_with_gaps(
    matched_indices: list[int],
    all_chunks: list[dict],
    fill_gap_indices: int = 2,
) -> list[dict]:
    """Merge adjacent matched chunks and fill small gaps.

    Args:
        matched_indices: Indices of matched chunks
        all_chunks: All chunks from the document
        fill_gap_indices: Maximum gap (in chunk indices) to fill

    Returns:
        List of reconstructed chunk dicts
    """
    if not matched_indices or not all_chunks:
        return []

    # Sort matched indices
    sorted_indices = sorted(set(matched_indices))

    # Group into contiguous ranges, filling small gaps
    groups = []
    current_group = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        gap = idx - current_group[-1] - 1
        if gap <= fill_gap_indices:
            # Fill the gap
            for fill_idx in range(current_group[-1] + 1, idx):
                current_group.append(fill_idx)
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]

    groups.append(current_group)

    # Build reconstructed chunks
    result = []
    for group in groups:
        chunks_in_group = [all_chunks[i] for i in group if i < len(all_chunks)]
        if not chunks_in_group:
            continue

        # Combine text from all chunks in group
        combined_text = "\n\n".join(c["text"] for c in chunks_in_group)
        start_char = chunks_in_group[0]["start_char"]
        end_char = chunks_in_group[-1]["end_char"]

        result.append(
            {
                "text": combined_text,
                "start_char": start_char,
                "end_char": end_char,
                "chunk_indices": group,
            }
        )

    return result
