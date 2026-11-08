"""Loading papers and splitting them into chunks.

Concepts:
- JSONL = one JSON object per line. Easy to append and stream; common for ML datasets.
- Chunking = splitting long text into smaller pieces, because (1) embedding models have a
  maximum input length and (2) a small, focused chunk matches a question better than a
  whole long document. Too small = loses context; too big = diluted meaning.
- Overlap = repeating a few words between neighbouring chunks so a sentence that is cut
  in half still appears complete in at least one chunk.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # e.g. "2511.01234-0"
    paper_id: str  # e.g. "2511.01234"
    title: str
    text: str  # the chunk text, with the title in front (see make_chunks)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into windows of `chunk_size` words that share `overlap` words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):  # last window reached the end
            break
    return pieces


def make_chunks(papers: list[dict], chunk_size: int = 300, overlap: int = 10) -> list[Chunk]:
    """Turn papers into chunks. The title is added to every chunk ("contextual chunking"),
    so a chunk from the middle of an abstract still knows which paper it belongs to."""
    overlap = min(overlap, chunk_size - 1)
    chunks = []
    for paper in papers:
        for i, piece in enumerate(split_words(paper["abstract"], chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper['id']}-{i}",
                    paper_id=paper["id"],
                    title=paper["title"],
                    text=f"{paper['title']}. {piece}",
                )
            )
    return chunks
