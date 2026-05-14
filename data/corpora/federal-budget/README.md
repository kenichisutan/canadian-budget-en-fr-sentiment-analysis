# Federal budget corpus (Canada)

Bilingual HTML snapshots and merged UTF-8 plain text from [budget.canada.ca](https://budget.canada.ca/home-accueil-fr.html), produced by `scripts/budget_corpus/`.

## Years covered

- **2015–2019, 2021–2025** (English and French).
- **2020 is omitted** by design: there is no consolidated Budget 2020 site section comparable to other years; the gap is documented in the scraper output.

## Layout

- `raw/{year}/fr|en/` — one `.html` file per fetched page.
- `raw/{year}/manifest-fr.json` / `manifest-en.json` — ordered list of `{ "url", "file" }` used for reproducible merges.
- `processed/{year}/` — `budget-{year}-{lang}.merged.html`, `budget-{year}-{lang}.txt`.

Merged outputs keep **main narrative only**: archival banners (`gc-archv`, `#archive_box`), site header/footer, Prev/TOC/Next rows, “On this page” sidebars, and leading “Archived …” title labels are removed. Plain text also drops short paragraphs that match standard Government of Canada archival / feedback boilerplate. Raw snapshots under `raw/` are unchanged.

## Regeneration

From the repository root (`canadian-budget-sentiment-analysis/`):

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Smoke test one year (recommended before a full 2015–2019,2021–2025 run):
python scripts/budget_corpus/cli.py download --years 2025 --delay 1.0
python scripts/budget_corpus/cli.py merge --years 2025
# Full corpus:
python scripts/budget_corpus/cli.py download --years 2015-2019,2021-2025
python scripts/budget_corpus/cli.py merge --years 2015-2019,2021-2025
```

Options: `--delay 1.0` (seconds between requests), `--no-separators` (omit `<<< source: ... >>>` lines in `.txt`).

## QA notes (2024 spot-check design)

After downloading **2024** (both languages):

- **TOC era URL filter:** only links whose path contains `/{year}/report-rapport/` and ends in `.html` are collected; paths containing `signaler-un-probleme`, `share-this-page`, or `/contact` are dropped. Links must match the target language in the filename (`-fr`/`-fra` vs `-en`/`-eng`, or the matching `home-accueil-*.html` home page) so French runs do not pull English chapter URLs and vice versa.
- **Crawl era (2016–2019):** the same filename language rule applies during BFS. Discovery **seeds** both the language home page and `/{year}/docs/plan/toc-tdm-{fr|en}.html`, because English homes often link only to the “brief” and not into the full plan. If you already downloaded with an older scraper, delete `raw/2016` (and other affected years) before re-running `download` so manifests and folders stay consistent.
- **2015 plan (HTML):** chapter files use **two** naming schemes on the archive: `ch1-fra.html`–`ch2-fra.html`, then `ch3-0-fra.html`–`ch5-0-fra.html` (and the English `-eng` counterparts); see the [2015 chapter 3 (FR)](https://budget.canada.ca/2015/docs/plan/ch3-0-fra.html) page. The scraper probes both patterns, then `anx{n}-fra|eng` annex pages and `toc-tdm-fra|eng.html` when present.
- **Sanity check:** run `wc -w data/corpora/federal-budget/processed/2024/budget-2024-*.txt` — merged word counts should be the same order of magnitude as the main budget PDF narrative (not identical, because annexes and tax detail pages differ).
- **Spot alignment:** open paired raw files with the same basename (e.g. `chap3-fr.html` / `chap3-en.html`) and compare a few paragraphs.

## Developer layout

Python modules live in [`scripts/budget_corpus/`](../../../scripts/budget_corpus/) (`discover.py`, `download.py`, `merge.py`, `extract_text.py`, `config.yaml`).

Content is Government of Canada material retrieved from budget.canada.ca for research; cite the official budget publications when publishing derived work.
