from __future__ import annotations

import json
import html as htmllib
from pathlib import Path
from budget_corpus.extract_text import extracted_main_inner_html, html_to_plain_text
from budget_corpus.paths import PROCESSED_ROOT, RAW_ROOT


def _load_manifest(year: int, lang: str) -> list[dict[str, str]]:
    p = RAW_ROOT / str(year) / f"manifest-{lang}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid manifest format: {p}")
    return data


def merge_year_lang(
    year: int,
    lang: str,
    *,
    separators: bool = True,
) -> tuple[Path, Path]:
    entries = _load_manifest(year, lang)
    year_raw = RAW_ROOT / str(year)

    merged_parts: list[str] = []
    text_parts: list[str] = []

    merged_parts.append(
        f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8"/>'
        f"<title>Budget {year} ({lang}) merged snapshot</title></head><body>\n"
    )

    for entry in entries:
        rel = entry["file"]
        fp = year_raw / rel
        if not fp.exists():
            raise FileNotFoundError(f"Expected raw file missing: {fp}")
        chunk = fp.read_text(encoding="utf-8", errors="surrogateescape")
        fname = Path(rel).name
        body_html = extracted_main_inner_html(chunk)
        merged_parts.append(f'<article data-source="{htmllib.escape(fname)}">\n')
        if body_html.strip():
            merged_parts.append(f'<div class="budget-extracted">\n{body_html}\n</div>\n')
        else:
            merged_parts.append(
                f'<!-- no main/#wb-cont region detected; raw page follows -->\n{chunk}\n'
            )
        merged_parts.append("\n</article>\n")

        plain = html_to_plain_text(chunk)
        if separators:
            text_parts.append(f"<<< source: {fname} >>>\n")
        if plain:
            text_parts.append(plain)
        if separators and plain:
            text_parts.append("\n")

    merged_parts.append("</body></html>\n")
    merged_html = "".join(merged_parts)
    merged_txt = "\n".join(text_parts).strip() + "\n"

    out_dir = PROCESSED_ROOT / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"budget-{year}-{lang}.merged.html"
    txt_path = out_dir / f"budget-{year}-{lang}.txt"
    html_path.write_text(merged_html, encoding="utf-8")
    txt_path.write_text(merged_txt, encoding="utf-8")
    return html_path, txt_path
