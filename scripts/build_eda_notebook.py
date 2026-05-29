#!/usr/bin/env python3
"""One-off generator for data/federal_budget_corpus_eda.ipynb — run from repo root."""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
        "execution_count": None,
    }


def main() -> None:
    cells: list[dict] = []

    cells.append(
        md(
            """# Federal Budget Corpus — Exploratory Analysis

Bilingual EN/FR corpus from [budget.canada.ca](https://budget.canada.ca) (2015–2019, 2021–2025; **2020 omitted**).

**Run from the repository root** so `scripts/budget_corpus` imports work:

```bash
pip install -r requirements.txt -r requirements-notebook.txt
jupyter notebook data/federal_budget_corpus_eda.ipynb
```

This notebook inventories raw and processed files and surfaces signals relevant to sentiment analysis and further text processing."""
        )
    )

    cells.append(
        code(
            """
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

NOTEBOOK_DIR = Path.cwd()
if (NOTEBOOK_DIR / "scripts" / "budget_corpus").is_dir():
    REPO_ROOT = NOTEBOOK_DIR
elif (NOTEBOOK_DIR.parent / "scripts" / "budget_corpus").is_dir():
    REPO_ROOT = NOTEBOOK_DIR.parent
else:
    raise RuntimeError("Run from repo root or data/ so scripts/budget_corpus is found")

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from budget_corpus.extract_text import extracted_main_inner_html, html_to_plain_text, normalize_whitespace
from budget_corpus.paths import CONFIG_PATH, PROCESSED_ROOT, RAW_ROOT

FIGURES_DIR = REPO_ROOT / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FIGURES = True


def savefig(name: str) -> None:
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / name, bbox_inches="tight", dpi=120)


with CONFIG_PATH.open(encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)
SKIP_YEARS = set(_cfg.get("skip_years", []))

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12})
LANG_COLORS = {"en": "#2E86AB", "fr": "#A23B72"}
"""
        )
    )

    cells.append(
        code(
            """
# Filename prefixes on budget.canada.ca (EN / FR share the same slug roots)
DOC_TYPE_PATTERNS = [
    (r"^chap", "chap"),       # chap1, chap2, … — main budget chapters (policy narrative)
    (r"^ch[0-9]", "chap"),    # ch1-eng, ch3-0-fra, … — 2015-era chapter naming
    (r"^anx", "anx"),         # anx1, … — annexes (tables, technical schedules)
    (r"^intro", "intro"),     # intro — budget introduction
    (r"^overview", "overview"),  # overview-apercu — high-level overview / aperçu
    (r"^foreword", "foreword"),  # foreword-avant-propos — ministerial foreword
    (r"^toc-tdm", "toc"),     # toc-tdm — table of contents (EN) / table des matières (FR)
    (r"^tm-mf", "tm-mf"),     # tm-mf — tax measures: supplementary information / mesures fiscales
    (r"^nwmm", "nwmm"),       # nwmm-amvm — notice of ways & means motion / avis de motion de voies et moyens
    (r"^gdql", "gdql"),       # gdql-egdqv — gender, diversity & inclusion impacts report / énoncé genre-diversité-inclusion
    (r"^p[1-4]-", "part"),    # p1, p2, … — budget “parts” (2021 site structure)
    (r"^reg-", "reg"),        # reg — draft regulatory amendments (e.g. GST/HST regulations)
]


def file_stem(fname: str) -> str:
    return re.sub(r"-(en|fr|eng|fra)\\.html$", "", fname, flags=re.I)


def classify_doc_type(fname: str) -> str:
    base = file_stem(Path(fname).name)
    for pat, label in DOC_TYPE_PATTERNS:
        if re.search(pat, base, re.I):
            return label
    return "other"


def word_count(text: str) -> int:
    return len(text.split())


def parse_source_markers(text: str) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    current_source = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^<<< source: (.+?) >>>\\s*$", line)
        if m:
            if current_source is not None:
                segments.append((current_source, word_count("\\n".join(buf))))
            current_source = m.group(1)
            buf = []
        else:
            buf.append(line)
    if current_source is not None:
        segments.append((current_source, word_count("\\n".join(buf))))
    return segments


def heuristic_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\\s+", normalize_whitespace(text))
    return [p.strip() for p in parts if p.strip()]


def digit_token_share(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if any(c.isdigit() for c in t)) / len(tokens)


ARCHIVED_RE = re.compile(r"\\barchived\\b|\\barchivé", re.I)
"""
        )
    )

    cells.append(
        code(
            """
def build_page_dataframe() -> pd.DataFrame:
    rows = []
    for year_dir in sorted(RAW_ROOT.iterdir(), key=lambda p: p.name):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for lang in ("en", "fr"):
            manifest_path = year_dir / f"manifest-{lang}.json"
            if not manifest_path.exists():
                continue
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in entries:
                rel = entry["file"]
                fp = year_dir / rel
                html = fp.read_text(encoding="utf-8", errors="replace")
                plain = html_to_plain_text(html)
                inner = extracted_main_inner_html(html)
                paras = [normalize_whitespace(p) for p in re.split(r"\\n{2,}", plain) if normalize_whitespace(p)]
                rows.append(
                    {
                        "year": year,
                        "lang": lang,
                        "file": rel,
                        "basename": Path(rel).name,
                        "url": entry.get("url", ""),
                        "stem": file_stem(Path(rel).name),
                        "doc_type": classify_doc_type(rel),
                        "chars": len(plain),
                        "words": word_count(plain),
                        "paragraphs": len(paras),
                        "has_archived_banner": bool(ARCHIVED_RE.search(plain)),
                        "extraction_empty": word_count(plain) < 100,
                        "inner_html_chars": len(inner),
                        "digit_share": digit_token_share(plain),
                        "plain_text": plain,
                    }
                )
    return pd.DataFrame(rows)


def build_merged_year_dataframe() -> pd.DataFrame:
    rows = []
    for year_dir in sorted(PROCESSED_ROOT.iterdir(), key=lambda p: p.name):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for lang in ("en", "fr"):
            path = year_dir / f"budget-{year}-{lang}.txt"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            segments = parse_source_markers(text)
            rows.append(
                {
                    "year": year,
                    "lang": lang,
                    "merged_words": word_count(text),
                    "merged_chars": len(text),
                    "source_blocks": len(segments),
                    "segment_words_sum": sum(w for _, w in segments),
                }
            )
    return pd.DataFrame(rows)


pages = build_page_dataframe()
merged_years = build_merged_year_dataframe()
print(f"Pages: {len(pages):,}  |  Years: {pages['year'].nunique()}  |  Merged files: {len(merged_years)}")
pages.head(3)
"""
        )
    )

    cells.append(md("## 1. Corpus inventory\n\nOverview of coverage, document mix, and total size by year and language."))

    cells.append(
        code(
            """
year_summary = (
    pages.groupby(["year", "lang"])
    .agg(pages=("basename", "count"), total_words=("words", "sum"), total_chars=("chars", "sum"))
    .reset_index()
)
pivot_words = year_summary.pivot(index="year", columns="lang", values="total_words")
pivot_words["fr_en_ratio"] = pivot_words["fr"] / pivot_words["en"]
display(year_summary.pivot(index="year", columns="lang", values="total_words").astype("Int64"))
display(pivot_words[["en", "fr", "fr_en_ratio"]].round(3))
if SKIP_YEARS:
    print("Skipped years (no budget site slice):", sorted(SKIP_YEARS))
"""
        )
    )

    cells.append(
        code(
            """
# Stacked bar: pages per year by doc_type — one full-size chart per language
page_counts = pages.groupby(["year", "lang", "doc_type"]).size().reset_index(name="n")
doc_types = sorted(page_counts["doc_type"].unique())
palette = sns.color_palette("tab20", n_colors=len(doc_types))
color_map = dict(zip(doc_types, palette))

for lang in ("en", "fr"):
    sub = page_counts[page_counts["lang"] == lang]
    pivot = sub.pivot(index="year", columns="doc_type", values="n").fillna(0).astype(int)
    pivot = pivot.reindex(columns=[c for c in doc_types if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(pivot))
    years = pivot.index.to_numpy()
    for doc_type in pivot.columns:
        vals = pivot[doc_type].to_numpy()
        ax.bar(years, vals, bottom=bottom, label=doc_type, color=color_map[doc_type])
        bottom = bottom + vals
    ax.set_xlabel("Year")
    ax.set_ylabel("Page count")
    ax.set_title(f"Pages per year by document type — {lang.upper()}")
    ax.legend(title="Document type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    savefig(f"01_pages_by_doc_type_{lang}.png")
    plt.show()

# Full report per year (total pages, no doc_type split)
report_pages = pages.groupby(["year", "lang"]).size().reset_index(name="n")
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(report_pages["year"].unique()))
years_sorted = sorted(report_pages["year"].unique())
w = 0.35
for i, lang in enumerate(("en", "fr")):
    sub = report_pages[report_pages["lang"] == lang].set_index("year").reindex(years_sorted)
    offset = -w / 2 if lang == "en" else w / 2
    ax.bar(x + offset, sub["n"], w, label=lang.upper(), color=LANG_COLORS[lang])
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years_sorted])
ax.set_xlabel("Year")
ax.set_ylabel("Total pages in corpus")
ax.set_title("Full report size per year (all sections combined)")
ax.legend()
plt.tight_layout()
savefig("01_report_pages_per_year.png")
plt.show()
report_pages.pivot(index="year", columns="lang", values="n")
"""
        )
    )

    cells.append(
        md(
            "**Reading:** The per-language stacked bars show section mix; the full-report chart counts every downloaded page per year (EN vs FR). Annex-heavy years (`anx`, `tm-mf`, `gdql`) add table-like content that may dominate token counts but carry less narrative sentiment than `chap` / `intro` sections."
        )
    )

    cells.append(
        code(
            """
# Line chart: total words by year (2020 gap visible on x-axis)
plot_df = year_summary.copy()
all_years = list(range(pages["year"].min(), pages["year"].max() + 1))
fig, ax = plt.subplots(figsize=(9, 4))
for lang in ("en", "fr"):
    sub = plot_df[plot_df["lang"] == lang].set_index("year").reindex(all_years)
    ax.plot(sub.index, sub["total_words"], marker="o", label=lang.upper(), color=LANG_COLORS[lang])
for y in SKIP_YEARS:
    if pages["year"].min() <= y <= pages["year"].max():
        ax.axvline(y, color="gray", ls="--", alpha=0.5)
        ax.text(y, ax.get_ylim()[1] * 0.95, "2020\\n(no site)", ha="center", fontsize=8, color="gray")
ax.set_xlabel("Year")
ax.set_ylabel("Total words (page sum)")
ax.set_title("Corpus size over time")
ax.legend()
plt.tight_layout()
savefig("02_words_timeline.png")
plt.show()
"""
        )
    )

    cells.append(
        code(
            """
# Grouped bar: EN vs FR words per year + FR/EN ratio labels
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(pivot_words))
w = 0.35
ax.bar(x - w / 2, pivot_words["en"], w, label="EN", color=LANG_COLORS["en"])
ax.bar(x + w / 2, pivot_words["fr"], w, label="FR", color=LANG_COLORS["fr"])
ax.set_xticks(x)
ax.set_xticklabels(pivot_words.index.astype(str))
for i, yr in enumerate(pivot_words.index):
    ratio = pivot_words.loc[yr, "fr_en_ratio"]
    ax.text(i, max(pivot_words.loc[yr, "en"], pivot_words.loc[yr, "fr"]) * 1.01, f"{ratio:.2f}", ha="center", fontsize=8)
ax.set_xlabel("Year")
ax.set_ylabel("Total words")
ax.set_title("EN vs FR volume per year (label = FR/EN ratio)")
ax.legend()
plt.tight_layout()
savefig("03_en_fr_words_by_year.png")
plt.show()
"""
        )
    )

    cells.append(
        md(
            "**Reading:** French budgets are consistently longer than English (~1.2–1.4×) across most years; 2015 is much smaller and structurally different."
        )
    )

    cells.append(md("## 2. Text structure (segmentation)\n\nHeuristic sentence splits — use spaCy or pySBD for production sentence boundary detection."))

    cells.append(
        code(
            """
struct_rows = []
for _, row in pages.iterrows():
    sents = heuristic_sentences(row["plain_text"])
    sent_lens = [word_count(s) for s in sents]
    paras = [normalize_whitespace(p) for p in re.split(r"\\n{2,}", row["plain_text"]) if normalize_whitespace(p)]
    para_lens = [word_count(p) for p in paras]
    struct_rows.append(
        {
            "year": row["year"],
            "lang": row["lang"],
            "doc_type": row["doc_type"],
            "stem": row["stem"],
            "n_sentences": len(sents),
            "mean_sent_len": np.mean(sent_lens) if sent_lens else np.nan,
            "median_sent_len": np.median(sent_lens) if sent_lens else np.nan,
            "mean_para_len": np.mean(para_lens) if para_lens else np.nan,
            "digit_share": row["digit_share"],
        }
    )
struct = pd.DataFrame(struct_rows)
struct.head(3)
"""
        )
    )

    cells.append(
        code(
            """
sample_years = [y for y in [2015, 2016, 2024, 2025] if y in struct["year"].values]
sub = struct[struct["year"].isin(sample_years)]
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=sub, x="year", y="median_sent_len", hue="lang", palette=LANG_COLORS, ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Median sentence length (words, heuristic)")
ax.set_title("Sentence length by year and language")
plt.tight_layout()
savefig("07_sentence_length.png")
plt.show()
"""
        )
    )

    cells.append(
        md(
            "### Sentence-length outliers\n\n"
            "Pages flagged when their **median sentence length** falls outside the boxplot whiskers "
            "(1.5× IQR within each year × language group shown above)."
        )
    )

    cells.append(
        code(
            """
def whisker_outliers(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    \"\"\"Mark rows outside Tukey whiskers (same rule as seaborn boxplot).\"\"\"
    parts = []
    for keys, grp in df.groupby(group_cols):
        q1 = grp[value_col].quantile(0.25)
        q3 = grp[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = grp[(grp[value_col] < lower) | (grp[value_col] > upper)].copy()
        if flagged.empty:
            continue
        flagged["whisker_low"] = lower
        flagged["whisker_high"] = upper
        flagged["direction"] = np.where(
            flagged[value_col] > upper,
            "high",
            np.where(flagged[value_col] < lower, "low", ""),
        )
        flagged["distance"] = np.where(
            flagged["direction"] == "high",
            flagged[value_col] - upper,
            lower - flagged[value_col],
        )
        parts.append(flagged)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def longest_sentence_preview(text: str, max_chars: int = 160) -> str:
    sents = heuristic_sentences(text)
    if not sents:
        return ""
    longest = max(sents, key=word_count)
    longest = " ".join(longest.split())
    return longest if len(longest) <= max_chars else longest[: max_chars - 3] + "..."


sent_outliers = whisker_outliers(sub, "median_sent_len", ["year", "lang"])
if sent_outliers.empty:
    print("No sentence-length outliers in the sampled years.")
else:
    # struct already has doc_type; only pull file paths from pages
    meta = pages[["year", "lang", "stem", "basename", "file"]].drop_duplicates()
    sent_outliers = sent_outliers.merge(meta, on=["year", "lang", "stem"], how="left")
    previews = []
    for _, row in sent_outliers.iterrows():
        match = pages[
            (pages.year == row["year"])
            & (pages.lang == row["lang"])
            & (pages.stem == row["stem"])
        ]
        text = match.iloc[0]["plain_text"] if not match.empty else ""
        previews.append(longest_sentence_preview(text))
    sent_outliers["longest_sentence_preview"] = previews

    show_cols = [
        "year",
        "lang",
        "basename",
        "doc_type",
        "median_sent_len",
        "direction",
        "distance",
        "whisker_low",
        "whisker_high",
        "longest_sentence_preview",
    ]
    out_table = sent_outliers[show_cols].sort_values(
        ["year", "lang", "distance"], ascending=[True, True, False]
    )
    print(f"Outlier pages: {len(out_table)}")
    display(out_table)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.stripplot(
        data=sub,
        x="year",
        y="median_sent_len",
        hue="lang",
        palette=LANG_COLORS,
        dodge=True,
        alpha=0.35,
        size=4,
        ax=ax,
    )
    for direction, marker, color in (("high", "^", "#c0392b"), ("low", "v", "#8e44ad")):
        pts = sent_outliers[sent_outliers["direction"] == direction]
        if pts.empty:
            continue
        ax.scatter(
            pts["year"],
            pts["median_sent_len"],
            marker=marker,
            s=90,
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=f"Outlier ({direction})",
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Median sentence length (words)")
    ax.set_title("Sentence-length outliers highlighted")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    savefig("07b_sentence_length_outliers.png")
    plt.show()
"""
        )
    )

    cells.append(
        code(
            """
fig, ax = plt.subplots(figsize=(9, 4))
order = ["chap", "intro", "overview", "anx", "tm-mf", "toc", "other"]
plot_types = [t for t in order if t in struct["doc_type"].unique()]
sns.violinplot(
    data=struct[struct["doc_type"].isin(plot_types)],
    x="doc_type",
    y="mean_para_len",
    hue="lang",
    order=plot_types,
    palette=LANG_COLORS,
    cut=0,
    ax=ax,
)
ax.set_xlabel("Document type")
ax.set_ylabel("Mean paragraph length (words)")
ax.set_title("Paragraph length by section type")
plt.tight_layout()
savefig("08_paragraph_by_doctype.png")
plt.show()
"""
        )
    )

    cells.append(
        code(
            """
digit_by_type = struct.groupby(["doc_type", "lang"])["digit_share"].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=digit_by_type, x="doc_type", y="digit_share", hue="lang", palette=LANG_COLORS, ax=ax)
ax.set_xlabel("Document type")
ax.set_ylabel("Mean share of tokens containing a digit")
ax.set_title("Numeric / table-like content proxy")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
savefig("09_digit_share_by_doctype.png")
plt.show()
"""
        )
    )

    cells.append(
        md(
            "**Reading:** High `digit_share` in `anx` and `tm-mf` suggests excluding or down-weighting those sections for narrative sentiment; use `chap` + `intro` + `overview` for comparable policy prose."
        )
    )

    cells.append(md("## 3. Sentiment-oriented lexicon probes\n\nNormalized keyword hit rates (not a sentiment model). Lists are easy to extend."))

    cells.append(
        code(
            """
LEXICONS = {
    "confidence": {
        "en": [r"\\bconfidence\\b", r"\\bconfident\\b", r"\\btrust\\b"],
        "fr": [r"\\bconfiance\\b", r"\\bconfiant\\b", r"\\bcrédibilité\\b"],
    },
    "uncertainty": {
        "en": [r"\\buncertain\\b", r"\\buncertainty\\b", r"\\bvolatile\\b"],
        "fr": [r"\\bincertitude\\b", r"\\bincertain\\b", r"\\bvolatil\\b"],
    },
    "risk": {
        "en": [r"\\brisk\\b", r"\\brisky\\b", r"\\bthreat\\b"],
        "fr": [r"\\brisque\\b", r"\\brisques\\b", r"\\bmenace\\b"],
    },
    "optimism": {
        "en": [r"\\boptimistic\\b", r"\\boptimism\\b", r"\\bopportunity\\b", r"\\bgrowth\\b"],
        "fr": [r"\\boptimiste\\b", r"\\boptimisme\\b", r"\\bopportunité\\b", r"\\bcroissance\\b"],
    },
    "fiscal_stress": {
        "en": [r"\\bdeficit\\b", r"\\bdebt\\b", r"\\bdeflation\\b"],
        "fr": [r"\\bdéficit\\b", r"\\bdette\\b", r"\\bdéflation\\b"],
    },
}


def lexicon_hits(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


lex_rows = []
for _, row in pages.iterrows():
    lang = row["lang"]
    for category, lang_patterns in LEXICONS.items():
        hits = lexicon_hits(row["plain_text"], lang_patterns[lang])
        lex_rows.append(
            {
                "year": row["year"],
                "lang": lang,
                "doc_type": row["doc_type"],
                "category": category,
                "hits": hits,
                "words": row["words"],
            }
        )
lex = pd.DataFrame(lex_rows)
lex["rate_per_10k"] = lex["hits"] / lex["words"].clip(lower=1) * 10_000
lex_year = lex.groupby(["year", "lang", "category"])["hits"].sum().reset_index()
lex_year_words = pages.groupby(["year", "lang"])["words"].sum().reset_index()
lex_year = lex_year.merge(lex_year_words, on=["year", "lang"])
lex_year["rate_per_10k"] = lex_year["hits"] / lex_year["words"].clip(lower=1) * 10_000
lex_year.head()
"""
        )
    )

    cells.append(
        code(
            """
# Heatmap: category x year (faceted by language)
for lang in ("en", "fr"):
    sub = lex_year[lex_year["lang"] == lang].pivot(index="category", columns="year", values="rate_per_10k")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(sub, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Lexicon hit rate per 10k words — {lang.upper()}")
    plt.tight_layout()
    savefig(f"10_lexicon_heatmap_{lang}.png")
    plt.show()
"""
        )
    )

    cells.append(
        code(
            """
# Line chart: uncertainty dimension over years
unc = lex_year[lex_year["category"] == "uncertainty"]
fig, ax = plt.subplots(figsize=(9, 4))
for lang in ("en", "fr"):
    sub = unc[unc["lang"] == lang]
    ax.plot(sub["year"], sub["rate_per_10k"], marker="o", label=lang.upper(), color=LANG_COLORS[lang])
ax.set_xlabel("Year")
ax.set_ylabel("Hits per 10k words")
ax.set_title("Uncertainty lexicon rate over time")
ax.legend()
plt.tight_layout()
savefig("11_uncertainty_timeline.png")
plt.show()
"""
        )
    )

    cells.append(
        md(
            "### Single-word EN/FR comparisons\n\n"
            "Each chart plots one English lemma on the EN corpus and one French lemma on the FR corpus "
            "(rates per 10,000 words by year). These are narrower than the multi-word lexicon categories above."
        )
    )

    cells.append(
        code(
            """
def single_word_rate_by_year(pattern_en: str, pattern_fr: str) -> pd.DataFrame:
    rows = []
    for year in sorted(pages["year"].unique()):
        for lang, pattern in (("en", pattern_en), ("fr", pattern_fr)):
            sub = pages[(pages.year == year) & (pages.lang == lang)]
            text = " ".join(sub["plain_text"])
            total_words = int(sub["words"].sum())
            hits = len(re.findall(pattern, text, flags=re.I))
            rows.append(
                {
                    "year": year,
                    "lang": lang,
                    "hits": hits,
                    "words": total_words,
                    "rate_per_10k": hits / max(total_words, 1) * 10_000,
                }
            )
    return pd.DataFrame(rows)


def plot_single_word_pair(
    word_en: str,
    word_fr: str,
    pattern_en: str,
    pattern_fr: str,
    fig_name: str,
) -> pd.DataFrame:
    rates = single_word_rate_by_year(pattern_en, pattern_fr)
    fig, ax = plt.subplots(figsize=(9, 4))
    en_sub = rates[rates["lang"] == "en"]
    fr_sub = rates[rates["lang"] == "fr"]
    ax.plot(
        en_sub["year"],
        en_sub["rate_per_10k"],
        marker="o",
        label=f"{word_en} (EN)",
        color=LANG_COLORS["en"],
    )
    ax.plot(
        fr_sub["year"],
        fr_sub["rate_per_10k"],
        marker="s",
        label=f"{word_fr} (FR)",
        color=LANG_COLORS["fr"],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Occurrences per 10,000 words")
    ax.set_title(f'"{word_en}" (English) vs "{word_fr}" (French)')
    ax.legend()
    plt.tight_layout()
    savefig(fig_name)
    plt.show()
    return rates


conf_rates = plot_single_word_pair(
    "confidence",
    "confiance",
    r"\\bconfidence\\b",
    r"\\bconfiance\\b",
    "11a_confidence_confiance.png",
)
display(conf_rates.pivot(index="year", columns="lang", values="rate_per_10k").round(2))

trust_rates = plot_single_word_pair(
    "trust",
    "crédibilité",
    r"\\btrust\\b",
    r"\\bcrédibilité\\b",
    "11b_trust_credibilite.png",
)
display(trust_rates.pivot(index="year", columns="lang", values="rate_per_10k").round(2))
"""
        )
    )

    cells.append(
        md(
            """### TMX sentence pairs (Budget 2025) — *confidence* / *trust* ↔ French

Aligned sentence pairs from [`budget-2025-en-budget-2025-fr.tmx`](corpora/federal-budget/budget-2025-en-budget-2025-fr.tmx) (LF Aligner). Below: EN sentences containing **confidence** or **trust**, with the matching FR segment — split into pairs where French uses the expected equivalent (**confiance** / **crédibilité**) vs. a different rendering.

*Trust* pairs that translate legal entities as *fiducie* are listed separately (not semantic “trust”)."""
        )
    )

    cells.append(
        code(
            """
import html
import xml.etree.ElementTree as ET

TMX_2025 = RAW_ROOT.parent / "budget-2025-en-budget-2025-fr.tmx"


def load_tmx_pairs(path: Path) -> pd.DataFrame:
    \"\"\"Parse LF Aligner TMX into EN/FR sentence pairs.\"\"\"
    rows = []
    for tu in ET.parse(path).iterfind(".//tu"):
        en_text = fr_text = None
        for tuv in tu.findall("tuv"):
            lang = (tuv.get("{http://www.w3.org/XML/1998/namespace}lang") or "").upper()
            seg = tuv.find("seg")
            text = html.unescape(seg.text or "") if seg is not None else ""
            if "EN" in lang:
                en_text = text.strip()
            elif "FR" in lang:
                fr_text = text.strip()
        if not en_text or not fr_text or en_text.startswith("<<<"):
            continue
        rows.append({"en": en_text, "fr": fr_text})
    return pd.DataFrame(rows)


def fr_match_type(fr: str, equivalent_pattern: str) -> str:
    return "equivalent" if re.search(equivalent_pattern, fr, flags=re.I) else "different"


def is_legal_entity_trust_pair(en: str, fr: str) -> bool:
    \"\"\"EN 'trust' as legal vehicle often becomes FR 'fiducie'.\"\"\"
    return bool(re.search(r"\\btrust\\b", en, re.I) and re.search(r"\\bfiducie", fr, re.I))


def filter_en_word(df: pd.DataFrame, en_pattern: str) -> pd.DataFrame:
    return df[df["en"].str.contains(en_pattern, case=False, regex=True)].copy()


def print_sentence_pairs(df: pd.DataFrame, heading: str, max_examples: int = 12) -> None:
    \"\"\"Print full EN/FR sentences as plain text (no truncated table cells).\"\"\"
    print(heading)
    if df.empty:
        print("  (none)\\n")
        return
    shown = df.head(max_examples)
    for i, row in enumerate(shown.itertuples(index=False), 1):
        print(f"\\n[{i}] EN: {row.en}")
        print(f"    FR: {row.fr}")
    if len(df) > len(shown):
        print(f"\\n  … and {len(df) - len(shown)} more.\\n")
    else:
        print()


def show_pair_tables(
    hits: pd.DataFrame,
    equivalent_pattern: str,
    en_label: str,
    fr_equiv_label: str,
    max_each: int = 10,
    diff_heading: str | None = None,
) -> None:
    hits = hits.copy()
    hits["fr_match"] = hits["fr"].apply(lambda t: fr_match_type(t, equivalent_pattern))
    n_eq = (hits["fr_match"] == "equivalent").sum()
    n_diff = (hits["fr_match"] == "different").sum()
    print(f"\\n{'=' * 72}")
    print(f"{en_label}  →  expected FR: {fr_equiv_label}")
    print(f"TMX segments: {len(hits)}  |  equivalent: {n_eq}  |  different: {n_diff}")

    equiv = hits[hits["fr_match"] == "equivalent"]
    diff = hits[hits["fr_match"] == "different"]
    if equiv.empty:
        print(f"\\n(no segments with {fr_equiv_label})")
    else:
        print_sentence_pairs(
            equiv, f"\\n--- Equivalent ({fr_equiv_label} in French) ---", max_each
        )
    if not diff.empty:
        heading = diff_heading or "\\n--- Different French rendering ---"
        print_sentence_pairs(diff, heading, max_each)


tmx_pairs = load_tmx_pairs(TMX_2025)
print(f"Loaded {len(tmx_pairs):,} aligned sentence pairs from {TMX_2025.name}")
"""
        )
    )

    cells.append(
        code(
            """
# --- confidence (EN) ↔ confiance (FR) ---
conf_hits = filter_en_word(tmx_pairs, r"\\bconfidence\\b")
show_pair_tables(
    conf_hits,
    equivalent_pattern=r"\\bconfiance\\b",
    en_label='"confidence"',
    fr_equiv_label="confiance",
    max_each=12,
    diff_heading="\\n--- confidence (EN) NOT rendered as confiance (FR) ---",
)

# Full list: every aligned segment with confidence but no confiance in French
conf_not_confiance = conf_hits[
    ~conf_hits["fr"].str.contains(r"\\bconfiance\\b", case=False, regex=True)
].copy()
print_sentence_pairs(
    conf_not_confiance,
    f"\\n{'=' * 72}\\n"
    f"Complete list: EN contains «confidence», FR has no «confiance» "
    f"({len(conf_not_confiance)} of {len(conf_hits)} segments)",
    max_examples=len(conf_not_confiance),
)
"""
        )
    )

    cells.append(
        code(
            """
# --- trust (EN) ↔ crédibilité (FR) — semantic uses (exclude legal fiducie) ---
trust_all = filter_en_word(tmx_pairs, r"\\btrust\\b")
trust_semantic = trust_all[~trust_all.apply(lambda r: is_legal_entity_trust_pair(r["en"], r["fr"]), axis=1)]

show_pair_tables(
    trust_semantic,
    equivalent_pattern=r"\\bcrédibilité\\b",
    en_label='"trust" (semantic, excl. fiducie/legal entity)',
    fr_equiv_label="crédibilité",
    max_each=12,
)

# Also show when FR uses confiance for EN trust (another common rendering)
trust_confiance = trust_semantic[trust_semantic["fr"].str.contains(r"\\bconfiance\\b", case=False, regex=True)]
if not trust_confiance.empty:
    print_sentence_pairs(
        trust_confiance,
        "\\n--- Semantic trust: French uses confiance (not crédibilité) ---",
        max_examples=12,
    )

legal_trust = trust_all[trust_all.apply(lambda r: is_legal_entity_trust_pair(r["en"], r["fr"]), axis=1)]
print_sentence_pairs(
    legal_trust,
    f"\\n--- Legal-entity trust → fiducie ({len(legal_trust)} pairs; excluded above) — sample ---",
    max_examples=6,
)
"""
        )
    )

    cells.append(
        code(
            """
# Top content words (2024) — minimal stopword list
STOP = {
    "the", "and", "of", "to", "in", "for", "a", "is", "that", "on", "with", "as", "by", "at", "an", "be", "this", "are", "or", "from",
    "le", "la", "les", "de", "des", "du", "et", "à", "en", "un", "une", "pour", "par", "sur", "dans", "est", "que", "qui", "au", "aux",
}


def top_tokens(lang: str, year: int = 2024, n: int = 20) -> pd.Series:
    text = " ".join(pages[(pages.year == year) & (pages.lang == lang)]["plain_text"])
    tokens = [re.sub(r"[^\\wàâäéèêëïîôùûüç'-]", "", t.lower()) for t in text.split()]
    tokens = [t for t in tokens if len(t) > 2 and t not in STOP]
    return pd.Series(dict(Counter(tokens).most_common(n)), name="count")


for lang in ("en", "fr"):
    print(f"\\nTop tokens — {lang.upper()} {2024}")
    display(top_tokens(lang))
"""
        )
    )

    cells.append(md("## 4. Extraction QA and processing recommendations"))

    cells.append(
        code(
            """
# Page words vs merged file words
page_totals = pages.groupby(["year", "lang"])["words"].sum().reset_index(name="page_words_sum")
qa = page_totals.merge(merged_years, on=["year", "lang"])
qa["diff_pct"] = (qa["merged_words"] - qa["page_words_sum"]) / qa["page_words_sum"].clip(lower=1) * 100
qa
"""
        )
    )

    cells.append(
        code(
            """
arch = pages.groupby(["year", "lang"])["has_archived_banner"].sum().reset_index(name="archived_pages")
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=arch, x="year", y="archived_pages", hue="lang", palette=LANG_COLORS, ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Pages with 'Archived' in plain text")
ax.set_title("Residual archival banner text after extraction")
plt.tight_layout()
savefig("12_archived_pages.png")
plt.show()
"""
        )
    )

    cells.append(
        code(
            """
problem = pages[(pages["extraction_empty"]) | (pages["inner_html_chars"] < 50)].copy()
problem = problem.sort_values(["words", "inner_html_chars"])
print(f"Low-content pages: {len(problem)}")
display(
    problem[["year", "lang", "basename", "words", "inner_html_chars", "doc_type"]].head(15)
)

high_digit = pages.nlargest(10, "digit_share")[["year", "lang", "basename", "doc_type", "digit_share", "words"]]
print("\\nHighest digit-token share (table-heavy):")
display(high_digit)
"""
        )
    )

    cells.append(
        md(
            """### Recommended next steps

1. **Analysis subset:** Use `doc_type` in `chap`, `intro`, `overview`, `foreword` for narrative sentiment; exclude or separate `toc`, `tm-mf`, and very high `digit_share` annex pages.
2. **Cleaning:** Strip residual “Archived …” title lines in `extract_text.py` (extend boilerplate list or remove leading h1 remnants).
3. **Segmentation:** Replace heuristic sentence splits with spaCy/pySBD before sentence-level sentiment or ABSA.
4. **Sentiment:** Lexicon probes here are exploratory; follow with multilingual transformers or aspect-based models per project abstract."""
        )
    )

    cells.append(
        code(
            """
import platform

print("Run completed:", datetime.now(timezone.utc).isoformat())
print("Python:", platform.python_version())
print("pandas:", pd.__version__)
print("Repo root:", REPO_ROOT)
print("Corpus raw:", RAW_ROOT)
print("Figures:", FIGURES_DIR if SAVE_FIGURES else "(not saved)")
"""
        )
    )

    import uuid

    for cell in cells:
        if "id" not in cell:
            cell["id"] = uuid.uuid4().hex[:8]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }

    out = Path(__file__).resolve().parents[1] / "data" / "federal_budget_corpus_eda.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
