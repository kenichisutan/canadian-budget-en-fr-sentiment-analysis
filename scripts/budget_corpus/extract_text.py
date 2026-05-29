from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


_BLOCK_TAGS = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote", "pre"}
)

# Paragraphs (or short runs) matching these are dropped from plain text after extraction.
_BOILERPLATE_SUBSTRINGS = (
    # English — archived banner / box
    "we have archived this page",
    "you can use it for research or reference",
    "archived information is provided for reference",
    "not subject to the government of canada web standards",
    "please contact us to request a format other than those available",
    "this web page has been archived on the web",
    # French
    "nous avons archivé cette page",
    "vous pouvez la consulter à des fins de recherche",
    "informations archivées",
    "elles ne sont pas assujetties aux normes web du gouvernement du canada",
    "pour obtenir ces informations dans un autre format",
    "l'information dont il est indiqué qu'elle est archivée",
    "l'information archivée",
    # Shared / navigation chrome that sometimes lands in body
    "skip to main content",
    "passer au contenu principal",
    "skip to about this site",
    "passer à « à propos de ce site »",
    "share this page",
    "partagez cette page",
    "date de modification",
    "feedback on this web site",
    "rétroaction sur ce site web",
    "report a problem on this page",
    "signaler un problème sur cette page",
)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines/tabs/NBSP) to single spaces."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    return " ".join(text.split())


def _strip_noise(soup: BeautifulSoup) -> None:
    for sel in ("script", "style", "nav", "header", "footer", "noscript"):
        for el in soup.find_all(sel):
            el.decompose()


def _strip_archival_and_chrome(soup: BeautifulSoup) -> None:
    """Remove GC archival banners, archive boxes, and page chrome not in header/footer."""
    for sel in (
        "section.gc-archv",
        ".gc-archv.wb-inview",
        "div#archive_box",
        "#archive_box",
        "aside.pagedetails",
        "section.pagedetails",
    ):
        for el in soup.select(sel):
            el.decompose()


def _main_root(soup: BeautifulSoup) -> Tag | None:
    """
    Prefer #wb-cont; if it is only an <h1>, use parent <main> so chapter body is included
    (budget.canada.ca often puts id=wb-cont on the title heading).
    """
    wb = soup.find(id="wb-cont")
    if wb and isinstance(wb, Tag):
        if wb.name.lower() == "h1":
            parent = wb.find_parent("main")
            if parent and isinstance(parent, Tag):
                return parent
        return wb
    node = soup.find(attrs={"role": "main"})
    if node and isinstance(node, Tag):
        return node
    body = soup.find("body")
    if body and isinstance(body, Tag):
        return body
    return soup if isinstance(soup, Tag) else soup.find()


def _strip_in_main_chrome(root: Tag) -> None:
    """Remove PDF download wells and decorative banners still inside main content."""
    for sel in ("a.gc-dwnld-lnk", ".well.gc-dwnld", "div.pageImg"):
        for el in root.select(sel):
            el.decompose()


def _strip_main_navigation_and_sidebar(root: Tag) -> None:
    """Remove in-page TOC sidebars and Prev/TOC/Next navigation rows."""
    for el in root.select("div.onThisPage"):
        el.decompose()
    for nav in root.select("nav.pagerNav"):
        nav.decompose()
    for div in list(root.select("div.row")):
        if not isinstance(div, Tag):
            continue
        t = div.get_text(" ", strip=True)
        if len(t) > 400:
            continue
        if "Previous" in t and "Next" in t and (
            "Table of Contents" in t or "Table des matières" in t
        ):
            div.decompose()


def _strip_archived_title_prefix(root: Tag) -> None:
    """Remove leading <small>… Archived …</small> inside the main h1."""
    h1 = root.find(id="wb-cont")
    if not h1 or not isinstance(h1, Tag) or h1.name.lower() != "h1":
        return
    for sm in h1.find_all("small", recursive=False):
        txt = sm.get_text(" ", strip=True).lower()
        if "archived" in txt or "archivé" in txt or "archivée" in txt:
            sm.decompose()


def _parse_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _normalize_tree_text_nodes(root: Tag) -> None:
    """Collapse runs of whitespace inside HTML text nodes (pretty-printed source)."""
    skip_parents = frozenset({"pre", "script", "style"})
    for node in list(root.find_all(string=True)):
        if isinstance(node, Comment):
            continue
        if not isinstance(node, NavigableString):
            continue
        parent = node.parent
        if parent and isinstance(parent, Tag) and parent.name.lower() in skip_parents:
            continue
        raw = str(node)
        normalized = normalize_whitespace(raw)
        if normalized != raw:
            node.replace_with(normalized)


def prepare_content_tree(html: str) -> Tag | None:
    """
    Parse HTML, remove site chrome and archival boilerplate, return the Tag subtree
    used for text extraction and for merged HTML fragments.
    """
    soup = _parse_soup(html)
    _strip_archival_and_chrome(soup)
    _strip_noise(soup)
    root = _main_root(soup)
    if not isinstance(root, Tag):
        return None
    _strip_in_main_chrome(root)
    _strip_main_navigation_and_sidebar(root)
    _strip_archived_title_prefix(root)
    _normalize_tree_text_nodes(root)
    return root


def extracted_main_inner_html(html: str) -> str:
    """Inner HTML of cleaned main (or wb-cont) region for lean merged documents."""
    root = prepare_content_tree(html)
    if root is None:
        return ""
    inner = root.decode_contents()
    return inner if inner.strip() else ""


def _visible_text_chunks(root: Tag) -> Iterable[str]:
    """Yield text segments from block-level-ish elements under root."""

    def walk(tag: Tag) -> Iterable[str]:
        for child in tag.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in _BLOCK_TAGS:
                t = normalize_whitespace(child.get_text(" ", strip=True))
                if t:
                    yield t
            elif name in ("ul", "ol", "table", "section", "article", "div", "main"):
                yield from walk(child)

    yield from walk(root)


def _drop_boilerplate_paragraphs(text: str) -> str:
    """Remove paragraph blocks that are mostly archival / site boilerplate."""
    parts: list[str] = []
    for block in re.split(r"\n{2,}", text.strip()):
        line = normalize_whitespace(block)
        low = line.lower()
        if not line:
            continue
        if len(line) < 500 and any(s in low for s in _BOILERPLATE_SUBSTRINGS):
            continue
        parts.append(line)
    out = "\n\n".join(parts)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def html_to_plain_text(html: str) -> str:
    """
    Convert budget HTML to UTF-8 plain text: drop chrome and archival notices,
    prefer real chapter body (main when wb-cont sits on h1), strip Prev/TOC/Next
    rows and on-this-page sidebars, then emit block text with blank lines.
    Each paragraph block is a single line (internal whitespace collapsed).
    """
    root = prepare_content_tree(html)
    if root is None:
        return ""
    chunks = list(_visible_text_chunks(root))
    text = "\n\n".join(chunks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _drop_boilerplate_paragraphs(text)
    return text.strip() + ("\n" if text.strip() else "")
