"""2本の実行結果 JSONL を課題ごとに対応付けて比較する（対応のある t 検定）。

`scripts/rank_both.py` と同じ統計だが、(1) `config` キーでも対応付けられる
（`bench_run` を直接叩いた出力にも使える）、(2) 任意個の variant を 1 つの基準に対して
横並びで表示する、の 2 点が違う。判定は np 主体で行う
（findings §11: 局所 np ↔ 本番 np r=0.938、局所 cog は 3 種すべて転移に失敗）。

使い方:
    python -m scripts.cmp_runs --base artifacts/bench_results/f1_hardmaxrun2.jsonl \\
        --var artifacts/bench_results/f1_stair0.30.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import os

METRICS = ("num_placed_items", "fill_score", "cog_score", "placement_proxy", "soft_proxy")
SCALE = {"num_placed_items": 100.0, "fill_score": 1.0, "cog_score": 1.0,
         "placement_proxy": 1.0, "soft_proxy": 1.0}
LABEL = {"num_placed_items": "np(pt)", "fill_score": "fill", "cog_score": "cog",
         "placement_proxy": "plc*", "soft_proxy": "soft*"}


def _load(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = os.path.basename(row.get("task_path") or row.get("config", ""))
            if key:
                rows[key] = row
    return rows


def _paired_t(deltas: list[float]) -> tuple[float, float, int]:
    """(mean, t, n) を返す。"""
    n = len(deltas)
    if n < 2:
        return (float("nan"), float("nan"), n)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else (float("inf") if mean else 0.0)
    return mean, t, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="基準の JSONL")
    parser.add_argument("--var", action="append", default=[], help="比較する JSONL（複数可）")
    args = parser.parse_args()

    base = _load(args.base)
    ok_base = {k: v for k, v in base.items() if v.get("status") == "success"}
    # placement/soft 代理は後から追加した項目なので、基準側に無ければ黙って落とす。
    metrics = [m for m in METRICS if all(m in r for r in ok_base.values())]
    print(f"base = {os.path.basename(args.base)}  (n={len(ok_base)})")
    for m in metrics:
        vals = [float(r[m]) * SCALE[m] for r in ok_base.values()]
        print(f"   {LABEL[m]:>8s} mean {sum(vals)/max(len(vals),1):8.3f}")
    print()
    hdr = f"{'variant':<22s} {'n':>3s}"
    for m in metrics:
        hdr += f" | {'Δ' + LABEL[m]:>9s} {'t':>6s}"
    hdr += f" | {'ng>0':>5s} {'pmax':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for path in args.var:
        var = _load(path)
        common = sorted(set(ok_base) & {k for k, v in var.items() if v.get("status") == "success"})
        line = f"{os.path.basename(path)[:22]:<22s} {len(common):>3d}"
        for m in metrics:
            d = [(float(var[k][m]) - float(ok_base[k][m])) * SCALE[m] for k in common]
            mean, t, _ = _paired_t(d)
            line += f" | {mean:>+9.2f} {t:>+6.2f}"
        n_ng = sum(1 for k in common if int(var[k].get("n_validator_ng", 0)) > 0)
        pmax = max((float(var[k].get("policy_time", 0.0)) for k in common), default=0.0)
        line += f" | {n_ng:>5d} {pmax:>6.2f}"
        print(line)


if __name__ == "__main__":
    main()
