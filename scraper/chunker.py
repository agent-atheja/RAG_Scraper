from config import CHUNK_OVERLAP, CHUNK_SIZE
from scraper.parser import ParsedPage

_CHAR_LIMIT = CHUNK_SIZE * 4


def _sliding_window(text: str, url: str, title: str, topic: str, heading: str) -> list[dict]:
    words = text.split()
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i: i + CHUNK_SIZE])
        chunks.append({
            "text": chunk_text,
            "url": url,
            "title": title,
            "topic": topic,
            "heading": heading,
            "has_code": "Code Example:" in chunk_text,
        })
    return chunks


def chunk_page(page: ParsedPage) -> list[dict]:
    chunks: list[dict] = []

    for section in page.sections:
        section_text = f"{section['heading']}\n{section['text']}"
        if section["code"]:
            section_text += f"\n\nCode Example:\n{section['code']}"

        if len(section_text) <= _CHAR_LIMIT:
            chunks.append({
                "text": section_text,
                "url": page.url,
                "title": page.title,
                "topic": page.topic,
                "heading": section["heading"],
                "has_code": bool(section["code"]),
            })
        else:
            chunks.extend(
                _sliding_window(
                    section_text, page.url, page.title, page.topic, section["heading"]
                )
            )

    return chunks
