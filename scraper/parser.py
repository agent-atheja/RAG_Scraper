from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag


@dataclass
class ParsedPage:
    url: str
    title: str
    topic: str
    sections: list[dict] = field(default_factory=list)
    raw_text: str = ""


def extract_topic(url: str) -> str:
    parts = url.replace("https://www.w3schools.com/", "").split("/")
    return parts[0] if parts and parts[0] else "general"


def _is_code_element(element: Tag) -> bool:
    if element.name == "pre":
        return True
    classes = element.get("class", [])
    return element.name == "div" and any(
        "w3-code" in c or "code" in c for c in classes
    )


def parse_page(url: str, html: str) -> ParsedPage | None:
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    main = soup.find("div", id="main") or soup.find("div", class_="w3-main")
    if not main or not isinstance(main, Tag):
        return None

    for tag in main.find_all(["script", "style", "button", "nav", "footer"]):
        tag.decompose()
    for tag in main.find_all(class_=["w3-bar", "w3-btn", "w3-example-link", "w3-green"]):
        tag.decompose()

    sections: list[dict] = []
    current_heading = title
    text_parts: list[str] = []
    code_parts: list[str] = []

    for element in main.find_all(["h1", "h2", "h3", "p", "pre", "div"]):
        if not isinstance(element, Tag):
            continue

        if element.name in ("h1", "h2", "h3"):
            if text_parts or code_parts:
                sections.append({
                    "heading": current_heading,
                    "text": " ".join(text_parts).strip(),
                    "code": "\n".join(code_parts).strip(),
                })
            current_heading = element.get_text(strip=True)
            text_parts = []
            code_parts = []

        elif element.name == "p":
            text = element.get_text(separator=" ", strip=True)
            if text:
                text_parts.append(text)

        elif _is_code_element(element):
            code = element.get_text(strip=True)
            if code:
                code_parts.append(code)

    if text_parts or code_parts:
        sections.append({
            "heading": current_heading,
            "text": " ".join(text_parts).strip(),
            "code": "\n".join(code_parts).strip(),
        })

    raw_text = " ".join(
        f"{s['heading']}. {s['text']} {s['code']}" for s in sections
    ).strip()

    return ParsedPage(
        url=url,
        title=title,
        topic=extract_topic(url),
        sections=sections,
        raw_text=raw_text,
    )
