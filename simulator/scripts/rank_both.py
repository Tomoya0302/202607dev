"""v38(既定) と search(GH_ORDER_SEARCH=1) を族別・対応ありで比較する（Phase 0 判定、
findings_2026-07.md §6.7「片族で有意に改善（t≥2）かつ他族で悪化しない（t>-2）」を適用）。

`scripts/eval_chunked.py` が書き出す2本の JSONL（v38 / search、同一タスク集合）を
`task_path`（=同一タスク）で対応付け、fill/cog/num_placed 各指標の対応のある差（search-v38）
について平均・SE・t値を算出する。scipy 不使用（大標本前提の正規近似、findings 同様）。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def _load(path: str) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["task_path"]] = row
    return rows


def _paired_t(deltas: list[float]) -> tuple[float, float, float, int]:
    """(mean, se, t, n) を返す（対応のある差の1標本t検定、母分散未知）。"""
    n = len(deltas)
    if n < 2:
        return (float("nan"),) * 3 + (n,)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    return mean, se, t, n


def compare(v38_path: str, search_path: str, label: str) -> dict:
    v38 = _load(v38_path)
    search = _load(search_path)
    common = sorted(set(v38) & set(search))
    skipped = sorted((set(v38) | set(search)) - set(common))
    metrics = ("fill_score", "num_placed_items", "cog_score")
    deltas: dict[str, list[float]] = defaultdict(list)
    n_ok = 0
    n_v38_fail = 0
    n_search_fail = 0
    for task in common:
        a, b = v38[task], search[task]
        if a.get("status") != "success":
            n_v38_fail += 1
            continue
        if b.get("status") != "success":
            n_search_fail += 1
            continue
        n_ok += 1
        for m in metrics:
            deltas[m].append(float(b[m]) - float(a[m]))

    print(f"\n=== {label} ===")
    print(f"  tasks: common={len(common)} usable_pairs={n_ok} "
          f"v38_fail={n_v38_fail} search_fail={n_search_fail} skipped(missing)={len(skipped)}")
    report = {"label": label, "n_pairs": n_ok, "metrics": {}}
    for m in metrics:
        mean, se, t, n = _paired_t(deltas[m])
        verdict = "improve(t>=2)" if t >= 2 else ("degrade(t<=-2)" if t <= -2 else "no-signal")
        print(f"  Δ{m}: mean={mean:+.4f} se={se:.4f} t={t:+.3f} n={n}  [{verdict}]")
        report["metrics"][m] = {"mean": mean, "se": se, "t": t, "n": n, "verdict": verdict}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-v38", required=True)
    parser.add_argument("--f1-search", required=True)
    parser.add_argument("--f2-v38", required=True)
    parser.add_argument("--f2-search", required=True)
    args = parser.parse_args()

    r1 = compare(args.f1_v38, args.f1_search, "family1 (1容器)")
    r2 = compare(args.f2_v38, args.f2_search, "family2 (2容器)")

    print("\n=== §6.7 判定（片族で t>=2 改善 かつ 他族で t>-2 非悪化）===")
    for m in ("fill_score", "num_placed_items", "cog_score"):
        t1 = r1["metrics"][m]["t"]
        t2 = r2["metrics"][m]["t"]
        robust = (t1 >= 2 and t2 > -2) or (t2 >= 2 and t1 > -2)
        print(f"  {m}: family1 t={t1:+.3f}, family2 t={t2:+.3f} -> {'ROBUST' if robust else 'not robust'}")


if __name__ == "__main__":
    main()
