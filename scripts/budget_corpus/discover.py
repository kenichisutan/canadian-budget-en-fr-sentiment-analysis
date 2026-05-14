from __future__ import annotations

import re
from collections import deque
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup

from budget_corpus.paths import CONFIG_PATH

ALLOWED_HOSTS = frozenset({"budget.canada.ca", "www.budget.canada.ca"})
DENY_PATH_FRAGMENTS = (
    "signaler-un-probleme",
    "share-this-page",
    "/contact",
    "javascript:",
    "mailto:",
)


def _basename_lang_marker(url: str) -> str | None:
    """
    Infer corpus language from the URL path basename (Government of Canada budget
    pages almost always use -fr/-fra vs -en/-eng before .html).
    Returns None if the marker is ambiguous or missing (caller should skip).
    """
    path = urlparse(url).path
    base = path.rsplit("/", 1)[-1].lower()
    if re.search(r"home-accueil-fr(?:\.html)?$", base):
        return "fr"
    if re.search(r"home-accueil-(?:en|eng)(?:\.html)?$", base):
        return "en"
    if re.search(r"-fra?\.html$", base):
        return "fr"
    if re.search(r"-eng?\.html$", base):
        return "en"
    return None


def _url_matches_corpus_lang(url: str, lang: str) -> bool:
    marker = _basename_lang_marker(url)
    if marker is None:
        return False
    return marker == lang


def _parse_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _canonical(url: str) -> str:
    u, _frag = urldefrag(url)
    parsed = urlparse(u)
    if parsed.query and "wbdisable" in parsed.query:
        # Drop wbdisable-only noise; keep other queries if any
        q = parsed.query
        parts = [p for p in q.split("&") if not p.startswith("wbdisable")]
        new_q = "&".join(parts)
        u = parsed._replace(query=new_q).geturl()
    return u.rstrip("/")


def _allowed_crawl_url(year: int, url: str, path_prefix: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    host = p.netloc.lower()
    if host and host not in ALLOWED_HOSTS:
        return False
    path = p.path or ""
    if not path.lower().endswith(".html"):
        return False
    if not path.startswith(path_prefix):
        return False
    low = path.lower()
    if "home-accueil" not in low and "/docs/" not in low:
        return False
    for frag in DENY_PATH_FRAGMENTS:
        if frag in low:
            return False
    return True


def _allowed_url(base: str, year: int, url: str, path_prefix: str | None, substrings: list[str]) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    host = p.netloc.lower()
    if host and host not in ALLOWED_HOSTS:
        return False
    path = p.path or ""
    if not path.lower().endswith(".html"):
        return False
    y = str(year)
    if f"/{y}/" not in path and not path.startswith(f"/{y}/"):
        return False
    if path_prefix and not path.startswith(path_prefix):
        return False
    if substrings and not any(s in path for s in substrings):
        return False
    low = path.lower()
    for frag in DENY_PATH_FRAGMENTS:
        if frag in low:
            return False
    return True


def _head_or_get_ok(client: httpx.Client, url: str) -> bool:
    try:
        r = client.head(url, follow_redirects=True)
        if r.status_code == 405 or r.status_code == 403:
            r = client.get(url, follow_redirects=True)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def discover_plan_chapters(
    client: httpx.Client, base_url: str, year: int, ycfg: dict[str, Any], lang: str
) -> list[str]:
    """
    Discover plan HTML URLs. Some years (e.g. 2015) mix chapter filenames:
    ch1-fra.html … ch2-fra.html, then ch3-0-fra.html … (see budget.canada.ca 2015 plan).
    Also pulls annex pages anx{n}-{suf}.html and the plan table-of-contents when present.
    """
    stem = ycfg["plan_prefix"]
    suf = ycfg["chapter_suffix_fr"] if lang == "fr" else ycfg["chapter_suffix_en"]
    max_n = int(ycfg.get("max_chapter_probe", 30))
    annex_max = int(ycfg.get("plan_annex_probe", 10))
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        c = _canonical(url)
        if c not in seen:
            seen.add(c)
            urls.append(c)

    # /2015/docs/plan/ch -> directory /2015/docs/plan/
    plan_dir: str | None = stem[:-2] if stem.endswith("ch") else None

    if plan_dir:
        toc_path = f"{plan_dir}toc-tdm-{suf}.html"
        toc_url = urljoin(base_url + "/", toc_path.lstrip("/"))
        if _head_or_get_ok(client, toc_url):
            add(toc_url)

    for n in range(1, max_n + 1):
        for slug in (f"{n}-{suf}", f"{n}-0-{suf}"):
            path = f"{stem}{slug}.html"
            url = urljoin(base_url + "/", path.lstrip("/"))
            if _head_or_get_ok(client, url):
                add(url)

    if plan_dir:
        for n in range(1, annex_max + 1):
            path = f"{plan_dir}anx{n}-{suf}.html"
            url = urljoin(base_url + "/", path.lstrip("/"))
            if _head_or_get_ok(client, url):
                add(url)

    return urls


def discover_toc_report(
    client: httpx.Client, base_url: str, year: int, ycfg: dict[str, Any], lang: str
) -> list[str]:
    key = "toc_fr" if lang == "fr" else "toc_en"
    toc_path = ycfg[key]
    toc_url = urljoin(base_url + "/", toc_path.lstrip("/"))
    r = client.get(toc_url, follow_redirects=True)
    r.raise_for_status()
    soup = _parse_soup(r.text)
    needle = f"/{year}/report-rapport/"
    seen: set[str] = set()
    ordered: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        full = urljoin(toc_url, href)
        full = _canonical(full)
        if needle not in full:
            continue
        if not full.lower().endswith(".html"):
            continue
        if not _allowed_url(base_url, year, full, f"/{year}/", ["/report-rapport/"]):
            continue
        if not _url_matches_corpus_lang(full, lang):
            continue
        if full not in seen:
            seen.add(full)
            ordered.append(full)
    return ordered


def discover_crawl(
    client: httpx.Client,
    base_url: str,
    year: int,
    ycfg: dict[str, Any],
    lang: str,
) -> list[str]:
    key = "home_fr" if lang == "fr" else "home_en"
    start_path = ycfg[key]
    start = urljoin(base_url + "/", start_path.lstrip("/"))
    # English (and sometimes French) landing pages link only to the "brief"; the full
    # plan lives under docs/plan/toc-tdm-*.html. Seed that TOC so BFS reaches chapters.
    toc_path = f"/{year}/docs/plan/toc-tdm-{lang}.html"
    toc_start = urljoin(base_url + "/", toc_path.lstrip("/"))
    path_prefix = ycfg["path_prefix"]
    max_pages = int(ycfg.get("max_pages", 200))

    seeds_unique: list[str] = []
    for s in (_canonical(start), _canonical(toc_start)):
        if s not in seeds_unique:
            seeds_unique.append(s)
    q: deque[str] = deque(seeds_unique)
    seen: set[str] = set()
    order: list[str] = []

    while q and len(order) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        if not _allowed_crawl_url(year, url, path_prefix):
            continue
        if not _url_matches_corpus_lang(url, lang):
            seen.add(url)
            continue
        seen.add(url)
        try:
            r = client.get(url, follow_redirects=True, timeout=60.0)
        except httpx.HTTPError:
            continue
        if r.status_code != 200:
            continue
        order.append(url)
        soup = _parse_soup(r.text)
        for a in soup.find_all("a", href=True):
            full = _canonical(urljoin(url, a["href"].strip()))
            if full in seen:
                continue
            if not _allowed_crawl_url(year, full, path_prefix):
                continue
            if not _url_matches_corpus_lang(full, lang):
                continue
            if full not in seen:
                q.append(full)
    return order


def discover_year_lang(
    client: httpx.Client, year: int, lang: str, cfg: dict[str, Any] | None = None
) -> list[str]:
    cfg = cfg or load_config()
    if year in cfg.get("skip_years", []):
        return []
    base = cfg["base_url"].rstrip("/")
    ykey = str(year)
    years = cfg.get("years", {})
    ycfg = years.get(ykey) or years.get(year)
    if ycfg is None:
        return []
    strategy = ycfg["strategy"]
    if strategy == "plan_chapters":
        return discover_plan_chapters(client, base, year, ycfg, lang)
    if strategy == "toc_report":
        return discover_toc_report(client, base, year, ycfg, lang)
    if strategy == "crawl_from_home":
        return discover_crawl(client, base, year, ycfg, lang)
    return []
