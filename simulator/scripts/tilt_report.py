"""`tilt_batch.py` の出力 JSONL を集計し、P1/P2/P3 の判定基準（plan §5）に照らして
サマリを表示する（診断専用、stdout のみ）。

使い方:
    python -m scripts.tilt_report --base artifacts/bench_results/tilt_f1_99_base.jsonl \
        [--hard-soft artifacts/bench_results/tilt_f1_99_hardsoft.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def _load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _paired_t(a: list[float], b: list[float]) -> tuple[float, float, int]:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0, n
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 1e-12
    t = mean / se
    return mean, t, n


def summarize_base(rows: list[dict], label: str) -> None:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    print(f"\n=== {label}: n={n} (errors={len(rows)-n}) ===")

    term_counts = defaultdict(int)
    for r in ok:
        term_counts[r.get("term_reason", "?")] += 1
    print("  term_reason:", dict(term_counts))

    hist_totals = defaultdict(int)
    for r in ok:
        for k, v in r.get("tilt_hist_all", {}).items():
            hist_totals[k] += v
    print("  tilt histogram (sum over tasks, per-item worst-tilt buckets):", dict(hist_totals))

    n_unsafe = sum(r.get("n_would_be_unsafe_if_gate_reapplied", 0) for r in ok)
    print(f"  n_would_be_unsafe_if_gate_reapplied (sum): {n_unsafe}")

    # 支持面別（soft/hard/floor/mixed）の平均・最大 tilt をタスク横断で集約
    support_vals: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        for kind, d in r.get("tilt_by_support_kind", {}).items():
            if d.get("n", 0) > 0:
                support_vals[kind].append(d["mean"])
    print("  mean-tilt-per-task by support_kind (avg of per-task means):")
    for kind, vals in support_vals.items():
        print(f"    {kind:8s} n_tasks={len(vals):3d}  mean={sum(vals)/len(vals):.3f}deg  max={max(vals):.3f}deg")

    soft_vals: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        for kind, d in r.get("tilt_by_is_soft", {}).items():
            if d.get("n", 0) > 0:
                soft_vals[kind].append(d["mean"])
    print("  mean-tilt-per-task by is_soft (of the tilting item itself):")
    for kind, vals in soft_vals.items():
        print(f"    {kind:8s} n_tasks={len(vals):3d}  mean={sum(vals)/len(vals):.3f}deg  max={max(vals):.3f}deg")

    n_cf_attempted = sum(1 for r in ok if r.get("counterfactual_transport", {}).get("attempted"))
    n_cf_pass = sum(1 for r in ok if r.get("counterfactual_transport", {}).get("would_pass_without_drift"))
    print(f"  counterfactual_transport: attempted={n_cf_attempted}/{n} would_pass_without_drift={n_cf_pass}")

    d_fill = [r["fill_score"] - r["fill_score_counterfactual_no_drift"] for r in ok if "fill_score" in r]
    if d_fill:
        print(f"  fill_score - fill_score_counterfactual_no_drift: mean={sum(d_fill)/len(d_fill):+.3f} "
              f"min={min(d_fill):+.3f} max={max(d_fill):+.3f}")


def compare(base_rows: list[dict], var_rows: list[dict], label: str) -> None:
    base_by = {r["task_path"]: r for r in base_rows if "error" not in r}
    var_by = {r["task_path"]: r for r in var_rows if "error" not in r}
    keys = sorted(set(base_by) & set(var_by))
    print(f"\n=== compare base vs {label}: n_paired={len(keys)} ===")
    metrics = ["fill_score", "num_placed_items", "cog_score"]
    for m in metrics:
        a = [var_by[k][m] for k in keys]
        b = [base_by[k][m] for k in keys]
        mean_d, t, n = _paired_t(a, b)
        scale = 100 if m == "num_placed_items" else 1
        print(f"  Δ{m:18s} mean={mean_d*scale:+.3f} t={t:+.2f} (n={n})")

    # tilt要約（平均tilt、by is_soft mean）の差
    def _mean_tilt(r: dict) -> float:
        h = r.get("tilt_by_is_soft", {})
        vals = [d["mean"] * d["n"] for d in h.values() if d.get("n", 0) > 0]
        ns = [d["n"] for d in h.values() if d.get("n", 0) > 0]
        return sum(vals) / sum(ns) if ns else 0.0

    a = [_mean_tilt(var_by[k]) for k in keys]
    b = [_mean_tilt(base_by[k]) for k in keys]
    mean_d, t, n = _paired_t(a, b)
    print(f"  Δmean_tilt_deg{'':7s} mean={mean_d:+.3f} t={t:+.2f} (n={n})")

    a2 = [var_by[k].get("n_tilt_events", 0) for k in keys]
    b2 = [base_by[k].get("n_tilt_events", 0) for k in keys]
    mean_d2, t2, n2 = _paired_t([float(x) for x in a2], [float(x) for x in b2])
    print(f"  Δn_tilt_events{'':7s} mean={mean_d2:+.3f} t={t2:+.2f} (n={n2})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, action="append",
                        help="baseline jsonl（複数指定可、族ごとに与える）")
    parser.add_argument("--hard-soft", action="append", default=[],
                        help="--hard-soft ablation jsonl（--base と同じ順序で対応させる）")
    args = parser.parse_args()

    for i, base_path in enumerate(args.base):
        rows = _load(base_path)
        summarize_base(rows, base_path)
        if i < len(args.hard_soft):
            hs_rows = _load(args.hard_soft[i])
            compare(rows, hs_rows, args.hard_soft[i])


if __name__ == "__main__":
    main()
