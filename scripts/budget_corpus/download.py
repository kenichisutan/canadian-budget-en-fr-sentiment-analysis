from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from budget_corpus.discover import discover_year_lang, load_config
from budget_corpus.paths import RAW_ROOT

USER_AGENT = (
    "budget-corpus-research/1.0 "
    "(+https://github.com/kenichisutan/canadian-budget-en-fr-sentiment-analysis; "
    "academic corpus snapshot of budget.canada.ca)"
)


def _safe_filename(url: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] or "index.html"
    name = unquote(name)
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "page.html"


def _write_manifest(year_dir: Path, lang: str, entries: list[dict[str, str]]) -> None:
    out = year_dir / f"manifest-{lang}.json"
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_year_lang(
    year: int,
    lang: str,
    *,
    delay: float = 1.0,
    cfg: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, str]]:
    cfg = cfg or load_config()
    if year in cfg.get("skip_years", []):
        print(f"SKIP year={year} (configured gap, e.g. no Budget 2020 site slice)")
        return []

    close_client = False
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=60.0,
            follow_redirects=True,
        )
        close_client = True
    try:
        urls = discover_year_lang(client, year, lang, cfg)
        if not urls:
            print(f"No URLs discovered for {year} {lang}")
            return []

        lang_dir = RAW_ROOT / str(year) / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        used_names: dict[str, int] = {}
        manifest: list[dict[str, str]] = []

        for url in urls:
            time.sleep(max(0.0, delay))
            last_exc: Exception | None = None
            text: str | None = None
            for attempt in range(4):
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    text = r.text
                    break
                except httpx.HTTPError as e:
                    last_exc = e
                    time.sleep(2**attempt * delay)
            if text is None:
                print(f"FAILED {url}: {last_exc}")
                continue

            base_name = _safe_filename(url)
            n = used_names.get(base_name, 0)
            used_names[base_name] = n + 1
            fname = base_name if n == 0 else f"{base_name.rsplit('.', 1)[0]}_{n + 1}.html"

            out_path = lang_dir / fname
            out_path.write_text(text, encoding="utf-8", errors="surrogateescape")
            manifest.append({"url": url, "file": f"{lang}/{fname}"})

        _write_manifest(RAW_ROOT / str(year), lang, manifest)
        print(f"Downloaded {len(manifest)} pages -> {lang_dir}")
        return manifest
    finally:
        if close_client:
            client.close()
