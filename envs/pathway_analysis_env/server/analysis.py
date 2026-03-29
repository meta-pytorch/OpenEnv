# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Differential expression (PyDESeq2) and over-representation analysis (ORA).

Counts matrices use **samples × genes** layout for PyDESeq2 ≥ 0.5.
"""

from __future__ import annotations

import io
import math
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    _PYDESQ2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy dep
    DeseqDataSet = None  # type: ignore[misc, assignment]
    DeseqStats = None  # type: ignore[misc, assignment]
    _PYDESQ2_AVAILABLE = False


def pydeseq2_available() -> bool:
    return _PYDESQ2_AVAILABLE


def default_analysis_options() -> Dict[str, Any]:
    """Defaults aligned with common RNA-seq practice (DESeq2 prefilter, directional ORA)."""
    return {
        "min_total_count": 10,
        "padj_alpha": 0.05,
        "ora_min_pathway_genes": 3,
        # Use "up" for treated-vs-control activation screens; "both" is the safe default.
        "de_query_direction": "both",
        "min_abs_log2fc": 0.0,
    }


def merge_analysis_options(case: Dict[str, Any]) -> Dict[str, Any]:
    out = default_analysis_options()
    raw = case.get("analysis_options")
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if k not in out or v is None:
            continue
        if k in ("min_total_count", "ora_min_pathway_genes"):
            out[k] = int(v)
        elif k in ("padj_alpha", "min_abs_log2fc"):
            out[k] = float(v)
        elif k == "de_query_direction":
            out[k] = str(v).lower().strip()
        else:
            out[k] = v
    return out


def filter_counts_by_minimum_total(
    counts_df: pd.DataFrame,
    min_total: int,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Remove genes with summed counts below ``min_total`` (DESeq2-style prefilter).

    Returns:
        (filtered_df, n_before, n_after)
    """
    if min_total <= 0:
        return counts_df, counts_df.shape[1], counts_df.shape[1]
    totals = counts_df.sum(axis=0)
    keep = totals >= min_total
    n_before = int(counts_df.shape[1])
    filtered = counts_df.loc[:, keep]
    n_after = int(filtered.shape[1])
    return filtered, n_before, n_after


def counts_dict_to_samples_by_genes(
    counts: Dict[str, Sequence[int]],
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    """Build a samples × genes count matrix from gene → per-sample counts."""
    sid_to_i = {sid: i for i, sid in enumerate(sample_ids)}
    rows = []
    for sid in sample_ids:
        j = sid_to_i[sid]
        rows.append([int(counts[g][j]) for g in counts])
    return pd.DataFrame(rows, index=list(sample_ids), columns=list(counts.keys()))


def build_sample_metadata(
    sample_ids: Sequence[str],
    condition_by_sample: Dict[str, str],
) -> pd.DataFrame:
    missing = [s for s in sample_ids if s not in condition_by_sample]
    if missing:
        raise ValueError(
            f"sample_metadata missing entries for sample_ids: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    conds = [condition_by_sample[s] for s in sample_ids]
    return pd.DataFrame({"condition": conds}, index=list(sample_ids))


def validate_counts_case(case: Dict[str, Any]) -> Optional[str]:
    """Return an error message if pipeline case JSON is inconsistent, else None."""
    counts = case.get("counts")
    sample_ids = case.get("sample_ids")
    if not isinstance(counts, dict) or not sample_ids:
        return None
    n = len(sample_ids)
    for gene, vals in counts.items():
        if len(vals) != n:
            return (
                f"Gene {gene!r} has {len(vals)} count values but "
                f"sample_ids has length {n}."
            )
    return None


def run_deseq2_contrast(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    alt_level: str,
    ref_level: str,
    *,
    padj_alpha: float = 0.05,
    min_replicates: int = 2,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Run PyDESeq2 Wald test for ``alt_level`` vs ``ref_level`` on column ``condition``.

    Returns:
        (de_rows, error_message). ``de_rows`` are sorted by ascending adjusted p-value.
    """
    if not _PYDESQ2_AVAILABLE:
        return [], "PyDESeq2 is not installed."

    levels = set(metadata_df["condition"].tolist())
    if ref_level not in levels or alt_level not in levels:
        return [], (
            f"Contrast invalid: need both reference {ref_level!r} and "
            f"alternate {alt_level!r} in sample metadata; got {sorted(levels)}."
        )

    try:
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata_df,
            design="~condition",
            refit_cooks=True,
            min_replicates=min_replicates,
            quiet=True,
        )
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition", alt_level, ref_level])
            stat_res.summary()
        res = stat_res.results_df
    except Exception as exc:  # pragma: no cover - fitting failures
        return [], f"DESeq2 failed: {exc}"

    de_rows: List[Dict[str, Any]] = []
    for gene, row in res.iterrows():
        padj = float(row["padj"]) if pd.notna(row["padj"]) else 1.0
        de_rows.append(
            {
                "gene": str(gene),
                "baseMean": float(row.get("baseMean", 0.0)),
                "log2FoldChange": float(row.get("log2FoldChange", 0.0)),
                "lfcSE": float(row.get("lfcSE", 0.0))
                if pd.notna(row.get("lfcSE"))
                else None,
                "pvalue": float(row.get("pvalue", 1.0))
                if pd.notna(row.get("pvalue"))
                else 1.0,
                "padj": padj,
                "significant": padj <= padj_alpha,
            }
        )
    de_rows.sort(key=lambda r: (r["padj"], -abs(r["log2FoldChange"])))
    return de_rows, None


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg FDR; returns q-values in original order (fallback)."""
    m = len(p_values)
    if m == 0:
        return []
    p_arr = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    adj_sorted = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(sorted_p[i] * m / (i + 1), running)
        adj_sorted[i] = running
    out = np.empty(m)
    out[order] = adj_sorted
    return np.clip(out, 0.0, 1.0).tolist()


def adjust_pvalues_bh(p_values: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg adjusted p-values using SciPy (preferred)."""
    m = len(p_values)
    if m == 0:
        return []
    p_arr = np.clip(
        np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0), 1e-300, 1.0
    )
    try:
        adj = false_discovery_control(p_arr, method="bh")
        return np.clip(adj, 0.0, 1.0).tolist()
    except Exception:
        return benjamini_hochberg(p_values)


def ora_fisher(
    de_genes: Sequence[str],
    pathway_genes: Dict[str, Sequence[str]],
    universe_genes: Sequence[str],
    *,
    min_pathway_genes: int = 3,
) -> List[Dict[str, Any]]:
    """
    Over-representation analysis (one-sided Fisher exact, greater overlap).

    ``universe_genes`` should be the **same gene set** used for DESeq2 (prefiltered).

    Pathways smaller than ``min_pathway_genes`` in the universe are skipped (reduces
    noise from tiny sets).
    """
    u: Set[str] = set(universe_genes)
    de: Set[str] = {g for g in de_genes if g in u}
    results: List[Dict[str, Any]] = []
    de_n = len(de)

    for pname, pgenes in pathway_genes.items():
        pset = {g for g in pgenes if g in u}
        if len(pset) < min_pathway_genes:
            continue
        overlap = sorted(de & pset)
        a = len(overlap)
        b = len(de - pset)
        c = len(pset - de)
        d = len(u) - a - b - c
        if d < 0:
            d = 0
        oddsr, p_raw = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_f = float(p_raw) if math.isfinite(float(p_raw)) else 1.0
        results.append(
            {
                "pathway": pname,
                "p_value": p_f,
                "odds_ratio": float(oddsr) if np.isfinite(oddsr) else None,
                "overlap_genes": overlap,
                "overlap_count": a,
                "pathway_size": len(pset),
                "de_in_universe": de_n,
                "gene_ratio": f"{a}/{len(pset)}",
            }
        )

    qvals = adjust_pvalues_bh([r["p_value"] for r in results])
    for r, q in zip(results, qvals):
        r["q_value"] = q
    results.sort(key=lambda x: (x["p_value"], -x["overlap_count"]))
    return results


def _safe_padj_value(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if math.isnan(x) else x


def pick_de_query_genes(
    de_rows: Sequence[Dict[str, Any]],
    *,
    padj_alpha: float = 0.05,
    max_genes: int = 200,
    direction: str = "both",
    min_abs_log2fc: float = 0.0,
) -> List[str]:
    """
    Genes for ORA query: significant by ``padj`` and optional **direction** (activation).

    ``direction``: ``\"up\"`` (alt > ref), ``\"down\"`` (alt < ref), or ``\"both\"``.
    """
    dir_norm = direction.lower().strip()
    if dir_norm not in ("up", "down", "both"):
        dir_norm = "both"

    def lfc_ok(r: Dict[str, Any]) -> bool:
        try:
            lfc = float(r.get("log2FoldChange", 0.0))
        except (TypeError, ValueError):
            return False
        if math.isnan(lfc):
            return False
        if dir_norm == "both":
            return abs(lfc) >= min_abs_log2fc
        if dir_norm == "up":
            return lfc >= min_abs_log2fc
        return lfc <= -min_abs_log2fc

    sig: List[str] = []
    for r in de_rows:
        if _safe_padj_value(r.get("padj", 1.0)) > padj_alpha:
            continue
        if not lfc_ok(r):
            continue
        sig.append(r["gene"])

    if not sig:
        for r in de_rows[:max_genes]:
            if lfc_ok(r):
                sig.append(r["gene"])
    if not sig:
        sig = [r["gene"] for r in de_rows[:max_genes]]
    return sig[:max_genes]


def compare_pathways_detail(
    pathway_a: str,
    pathway_b: str,
    pathway_genes: Dict[str, Sequence[str]],
    de_genes: Sequence[str],
) -> Dict[str, Any]:
    """Exclusive vs shared DE support between two pathways."""
    pa = set(pathway_genes.get(pathway_a, []))
    pb = set(pathway_genes.get(pathway_b, []))
    de = set(de_genes)
    only_a = sorted((pa - pb) & de)
    only_b = sorted((pb - pa) & de)
    shared = sorted((pa & pb) & de)
    return {
        "pathway_a": pathway_a,
        "pathway_b": pathway_b,
        "exclusive_to_a": only_a,
        "exclusive_to_b": only_b,
        "shared_de_support": shared,
        "pathway_a_size": len(pa),
        "pathway_b_size": len(pb),
        "overlap_pathway_genes": sorted(pa & pb),
    }


def overlap_genes_across_top_pathways(
    ora_rows: Sequence[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """DE genes that appear in more than one of the top-k pathways by p-value."""
    top = [r for r in ora_rows[:top_k] if r.get("overlap_genes")]
    gene_to_paths: Dict[str, List[str]] = {}
    for row in top:
        p = row["pathway"]
        for g in row.get("overlap_genes", []):
            gene_to_paths.setdefault(g, []).append(p)
    multi = {g: paths for g, paths in gene_to_paths.items() if len(paths) > 1}
    return {
        "genes_supporting_multiple_top_pathways": sorted(multi.keys()),
        "gene_to_pathways": {g: multi[g] for g in sorted(multi)},
    }


def top_hits_statistically_close(
    ora_rows: Sequence[Dict[str, Any]],
    *,
    ratio_threshold: float = 10.0,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Flag when the top two enriched pathways have similar p-values (ratio bound)."""
    if len(ora_rows) < 2:
        return {
            "close_top_hits": False,
            "p_ratio": None,
            "note": "fewer than 2 pathways",
        }
    p1 = ora_rows[0]["p_value"]
    p2 = ora_rows[1]["p_value"]
    if p1 <= 0 or p2 <= 0:
        ratio = None
        close = False
    else:
        ratio = max(p1, p2) / min(p1, p2)
        close = ratio <= ratio_threshold
    return {
        "close_top_hits": close,
        "p_ratio": ratio,
        "p_top1": p1,
        "p_top2": p2,
    }
