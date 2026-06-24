"""Load the knowledge base and split it into chunks.

This is the unglamorous half of RAG that decides most of your quality. Two ideas:

1. **A document is too big to retrieve as a unit.** If a customer asks about the
   return window, you want the 2 sentences about the return window — not the entire
   returns.md file. So we split documents into smaller **chunks** and retrieve those.

2. **Chunking is a tradeoff, not a setting you get "right".**
   - Chunks too BIG  -> each retrieved chunk carries lots of irrelevant text. That's
     noise for the model and wasted tokens (cost).
   - Chunks too SMALL -> a single chunk loses the surrounding context needed to make
     sense of it, and the answer may be split across chunks you don't both retrieve.
   The sweet spot depends on your content. For policy docs, splitting on the natural
   section headings (one topic per chunk) is a strong, simple strategy.

We chunk by markdown `##` section, and only fall back to fixed-size windows (with
overlap) if a section is unusually long. Overlap means adjacent windows share some
text, so a fact sitting on a window boundary still lands wholly inside one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of knowledge.

    `source` is the filename (e.g. "returns.md"). We use it as the unit of ground
    truth in evaluation (evaluation.py) — a question's "correct" answer lives in a
    known source doc, and a retrieval is a hit if a chunk from that doc shows up.
    """

    id: str
    source: str          # filename, e.g. "returns.md"
    doc_title: str        # the "# Title" of the file
    section: str          # the "## Section" heading this chunk came from
    text: str             # the chunk content (what we embed and feed the model)

    def for_embedding(self) -> str:
        """The text we actually embed.

        Prefixing the section heading gives the embedder topical context, which
        measurably improves retrieval for short chunks ("Return window" + body
        embeds closer to a return-window question than the bare body does).
        """
        return f"{self.doc_title} — {self.section}\n{self.text}"


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    sections: list[tuple[str, str]]  # (section_heading, section_body)


def load_knowledge_base(directory: str | Path) -> list[Document]:
    """Read every .md file in `directory` and parse it into titled sections."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {directory.resolve()}"
        )

    docs: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = _first_heading(raw) or path.stem
        sections = _split_sections(raw)
        docs.append(Document(source=path.name, title=title, sections=sections))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {directory.resolve()}")
    return docs


def chunk_documents(
    docs: list[Document], *, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Turn documents into retrievable chunks (one per section, split if too long)."""
    chunks: list[Chunk] = []
    for doc in docs:
        for section_heading, body in doc.sections:
            for piece in _window(body, max_chars=max_chars, overlap=overlap_chars):
                chunk_id = f"{doc.source}#{len(chunks)}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source=doc.source,
                        doc_title=doc.title,
                        section=section_heading,
                        text=piece.strip(),
                    )
                )
    return chunks


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #
def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (## heading, body) pairs."""
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue  # the doc title; handled separately
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(h, b) for h, b in sections if b]


def _window(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, preferring paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    windows: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                windows.append(buffer)
            # carry the tail of the previous window forward as overlap
            tail = buffer[-overlap:] if overlap and buffer else ""
            buffer = f"{tail}\n\n{para}" if tail else para
    if buffer:
        windows.append(buffer)
    return windows
