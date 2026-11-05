from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[a-z]{3,}")


def _normalize_condition_label(text: str) -> str:
    """Lowercase alphanumeric normalization for fuzzy condition matching."""
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


DEFAULT_TEXT_COLUMNS: tuple[str, ...] = (
    "description_short",
    "concise_description",
    "description",
    "dataset_desc",
)

STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "case",
    "cases",
    "close",
    "closeup",
    "collection",
    "condition",
    "conditions",
    "dataset",
    "detail",
    "detailed",
    "depicts",
    "data",
    "example",
    "featuring",
    "feature",
    "features",
    "for",
    "from",
    "general",
    "has",
    "have",
    "head",
    "illustrates",
    "image",
    "images",
    "in",
    "include",
    "includes",
    "including",
    "is",
    "it",
    "its",
    "left",
    "lesion",
    "lesions",
    "location",
    "locations",
    "lower",
    "male",
    "man",
    "multiple",
    "of",
    "old",
    "on",
    "over",
    "patient",
    "patients",
    "photo",
    "photograph",
    "photography",
    "picture",
    "posterior",
    "region",
    "regions",
    "right",
    "shows",
    "showing",
    "skin",
    "subject",
    "subjects",
    "such",
    "taken",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "tile",
    "tiles",
    "to",
    "up",
    "various",
    "view",
    "views",
    "was",
    "were",
    "with",
    "woman",
    "year",
    "years",
}


def _select_text_series(
    meta: pd.DataFrame, text_columns: Sequence[str] | None
) -> pd.Series | None:
    if text_columns is None:
        text_columns = DEFAULT_TEXT_COLUMNS
    text_series: pd.Series | None = None
    for col in text_columns:
        if col not in meta.columns:
            continue
        col_series = meta[col].fillna("").astype(str)
        if text_series is None:
            text_series = col_series
        else:
            mask = text_series.str.strip().astype(bool)
            text_series = text_series.where(mask, col_series)
    return text_series


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    tokens = TOKEN_PATTERN.findall(text)
    return [tok for tok in tokens if tok not in STOPWORDS]


def _tokenize_with_bigrams(text: str) -> list[str]:
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return tokens
    bigrams = [
        f"{a} {b}"
        for a, b in zip(tokens[:-1], tokens[1:])
        if a not in STOPWORDS and b not in STOPWORDS
    ]
    return tokens + bigrams


def compute_keyword_trends(
    meta: pd.DataFrame,
    year_col: str,
    text_columns: Sequence[str] | None = None,
    *,
    max_terms_per_year: int = 4,
    min_tokens_per_year: int = 40,
    min_term_count: int = 3,
) -> tuple[pd.DataFrame, "OrderedDict[int, list[str]]"]:
    if year_col not in meta.columns:
        return pd.DataFrame(), OrderedDict()
    text_series = _select_text_series(meta, text_columns)
    if text_series is None:
        return pd.DataFrame(), OrderedDict()

    df = pd.DataFrame({"year": meta[year_col], "text": text_series})
    df = df[df["text"].astype(str).str.strip().astype(bool)]
    df = df[df["year"].notna()]
    if df.empty:
        return pd.DataFrame(), OrderedDict()
    df["year"] = df["year"].round().astype(int)

    year_stats: dict[int, tuple[Counter[str], int]] = {}
    for year, group in df.groupby("year"):
        counter: Counter[str] = Counter()
        for text in group["text"]:
            counter.update(_tokenize_with_bigrams(text))
        total = int(sum(counter.values()))
        if total < min_tokens_per_year:
            continue
        year_stats[year] = (counter, total)

    if len(year_stats) < 2:
        return pd.DataFrame(), OrderedDict()

    df_counts: Counter[str] = Counter()
    for counter, _ in year_stats.values():
        for term, count in counter.items():
            if count >= min_term_count:
                df_counts[term] += 1

    n_years = len(year_stats)
    rows = []
    summary: OrderedDict[int, list[str]] = OrderedDict()
    for year in sorted(year_stats.keys()):
        counter, total = year_stats[year]
        scored_terms: list[tuple[str, float, float, int]] = []
        for term, count in counter.items():
            if count < min_term_count:
                continue
            doc_freq = df_counts.get(term, 1)
            idf = math.log((1 + n_years) / (1 + doc_freq)) + 1.0
            share = count / total if total else 0.0
            tfidf = share * idf
            scored_terms.append((term, tfidf, share, count))

        scored_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = scored_terms[:max_terms_per_year]
        summary[year] = [term for term, *_ in top_terms]
        for rank, (term, tfidf, share, count) in enumerate(top_terms, start=1):
            rows.append(
                {
                    "year": year,
                    "term": term,
                    "rank": rank,
                    "tfidf": tfidf,
                    "share": share,
                    "count": count,
                }
            )

    keyword_df = pd.DataFrame(rows)
    return keyword_df, summary


def plot_keyword_trends(
    keyword_df: pd.DataFrame,
    summary_by_year: Mapping[int, Sequence[str]] | None = None,
    *,
    title: str = "Distinctive dataset keywords by year",
    cmap: str = "plasma",
    figsize: tuple[float, float] = (3.54, 2.8),
    min_marker_size: float = 80.0,
    size_scale: float = 8000.0,
) -> tuple[plt.Figure, plt.Axes]:
    if keyword_df.empty:
        raise ValueError("keyword_df is empty; nothing to plot.")

    years = sorted(keyword_df["year"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    tfidf_vals = keyword_df["tfidf"].to_numpy()
    norm = colors.Normalize(vmin=tfidf_vals.min(), vmax=tfidf_vals.max())
    cmap_obj = cm.get_cmap(cmap)

    for row in keyword_df.itertuples(index=False):
        size = min_marker_size + size_scale * float(row.share)
        color = cmap_obj(norm(row.tfidf))
        ax.scatter(
            row.year,
            row.rank,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            zorder=2,
        )
        ax.text(
            row.year,
            row.rank,
            row.term,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if norm(row.tfidf) > 0.5 else "black",
            zorder=3,
        )

    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    max_rank = int(keyword_df["rank"].max())
    ax.set_ylim(max_rank + 0.6, 0.4)
    ax.set_ylabel("Rank within year (1 = most distinctive)")
    ax.set_xlabel("Release year")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("TF-IDF distinctiveness")

    if summary_by_year:
        latest_year = max(summary_by_year.keys())
        highlights = ", ".join(summary_by_year.get(latest_year, [])[:3])
        if highlights:
            ax.text(
                0.01,
                1.05,
                f"Latest ({latest_year}): {highlights}",
                transform=ax.transAxes,
                fontsize=7,
                color="#444",
            )

    fig.tight_layout()
    return fig, ax


def compute_condition_trends(
    meta: pd.DataFrame,
    *,
    year_col: str,
    condition_col: str,
    min_samples_per_year: int = 100,
    min_total_per_condition: int = 150,
    top_n: int = 8,
    condition_map: Mapping[str, str] | None = None,
    drop_conditions: Sequence[str] | None = None,
    min_years_per_condition: int = 3,
    min_share_threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    if year_col not in meta.columns or condition_col not in meta.columns:
        return pd.DataFrame(), {}

    df = meta[[year_col, condition_col]].copy()
    df = df.dropna()
    if df.empty:
        return pd.DataFrame(), {}

    df[year_col] = df[year_col].round().astype(int)
    df[condition_col] = df[condition_col].astype(str).str.strip()
    if condition_map:

        def _map_condition(label: str) -> str:
            normalized = _normalize_condition_label(label)
            if label in condition_map:
                return condition_map[label]
            if normalized in condition_map:
                return condition_map[normalized]
            return label

        df[condition_col] = df[condition_col].map(_map_condition)
    df = df[df[condition_col] != ""]
    if drop_conditions:
        drop_lookup = {_normalize_condition_label(c) for c in drop_conditions}
        df = df[
            ~df[condition_col]
            .map(lambda x: _normalize_condition_label(x))
            .isin(drop_lookup)
        ]
    if df.empty:
        return pd.DataFrame(), {}

    year_counts = df.groupby(year_col).size()
    valid_years = year_counts[year_counts >= min_samples_per_year].index
    df = df[df[year_col].isin(valid_years)]
    if df.empty or len(valid_years) < 2:
        return pd.DataFrame(), {}

    cond_totals = df.groupby(condition_col).size().sort_values(ascending=False)
    cond_keep = cond_totals[cond_totals >= min_total_per_condition]
    if cond_keep.empty:
        cond_keep = cond_totals
    keep_conditions = cond_keep.index.tolist()
    if not keep_conditions:
        return pd.DataFrame(), {}

    counts = (
        df[df[condition_col].isin(keep_conditions)]
        .groupby([year_col, condition_col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    year_totals = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.div(year_totals, axis=0).fillna(0.0)

    mean_share = shares.mean(axis=0)
    selected = mean_share.sort_values(ascending=False).head(top_n).index.tolist()
    min_years = max(1, int(min_years_per_condition))
    support = (shares[selected] >= float(min_share_threshold)).sum(axis=0)
    selected = support[support >= min_years].index.tolist()
    if not selected:
        return pd.DataFrame(), {}
    shares = shares[selected]

    summary = {"year_counts": year_counts.loc[shares.index]}  # align index
    last_year = shares.index.max()
    first_year = shares.index.min()
    summary["latest_top"] = shares.loc[last_year].sort_values(ascending=False)
    summary["fastest_risers"] = (
        (shares.loc[last_year] - shares.loc[first_year])
        .sort_values(ascending=False)
        .head(min(5, len(selected)))
    )
    summary["latest_year"] = int(last_year)
    summary["first_year"] = int(first_year)
    return shares, summary


def plot_condition_trends(
    share_df: pd.DataFrame,
    *,
    title: str = "Condition prevalence over time",
    figsize: tuple[float, float] = (3.54, 2.6),
    highlight_top: int = 3,
    palette: str = "tab10",
) -> tuple[plt.Figure, plt.Axes]:
    if share_df.empty:
        raise ValueError("share_df is empty; nothing to plot.")

    years = share_df.index.to_numpy()
    latest_order = share_df.loc[years[-1]].sort_values(ascending=False).index.tolist()
    colors_cycle = plt.get_cmap(palette)
    fig, ax = plt.subplots(figsize=figsize)

    for idx, condition in enumerate(latest_order):
        y = share_df[condition].to_numpy()
        color = colors_cycle(idx % colors_cycle.N)
        is_highlight = idx < highlight_top
        ax.plot(
            years,
            y,
            color=color,
            linewidth=2.2 if is_highlight else 1.0,
            alpha=0.95 if is_highlight else 0.35,
            solid_capstyle="round",
            label=condition,
        )
        ax.fill_between(
            years,
            0,
            y,
            color=color,
            alpha=0.12 if is_highlight else 0.03,
        )

    ax.set_xlabel("Release year")
    ax.set_ylabel("Share of condition volume")
    ax.set_title(title)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylim(0, min(0.65, share_df.values.max() * 1.1 + 0.03))
    ax.grid(axis="y", linestyle=(0, (2, 4)), linewidth=0.4, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    last_year = years[-1]
    first_year = years[0]
    for idx, condition in enumerate(latest_order[:highlight_top]):
        color = colors_cycle(idx % colors_cycle.N)
        start_share = share_df.loc[first_year, condition]
        end_share = share_df.loc[last_year, condition]
        ax.scatter(
            [first_year, last_year],
            [start_share, end_share],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            last_year + 0.2,
            end_share,
            f"{condition}  {end_share*100:.1f}%",
            fontsize=7,
            color="#222",
            va="center",
            ha="left",
        )

    handles = [
        plt.Line2D([0], [0], color=colors_cycle(i % colors_cycle.N), lw=2.2)
        for i in range(min(len(latest_order), 6))
    ]
    labels = latest_order[: len(handles)]
    ax.legend(
        handles,
        labels,
        fontsize=6.5,
        loc="upper left",
        frameon=False,
        title="Top conditions",
        title_fontsize=7,
    )

    fig.tight_layout()
    return fig, ax


DEFAULT_CLUSTER_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "hierarchy_final_all_levels.csv"
)


def load_condition_cluster_mapping(
    path: str | Path = DEFAULT_CLUSTER_PATH,
    *,
    level: str = "level_0.030",
) -> dict[str, str]:
    full_path = Path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Condition cluster mapping not found: {full_path}")
    df = pd.read_csv(full_path)
    level_key = level
    mapping = {}
    subset = df[df["level"] == level_key]
    if subset.empty:
        raise ValueError(f"No entries for level '{level_key}' in {full_path}")
    for condition, cluster in zip(subset["condition"], subset["cluster_name"]):
        cluster_name = (
            cluster if isinstance(cluster, str) and cluster.strip() else condition
        )
        condition_str = str(condition)
        mapping[condition_str] = cluster_name
        normalized = _normalize_condition_label(condition_str)
        if normalized:
            mapping[normalized] = cluster_name
    return mapping


__all__ = [
    "compute_keyword_trends",
    "plot_keyword_trends",
    "compute_condition_trends",
    "plot_condition_trends",
    "load_condition_cluster_mapping",
]
