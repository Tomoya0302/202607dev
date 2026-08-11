"""soft-on-soft / priority shift スクリーニング専用の比較（`cmp_runs.py`を拡張、
equilibrium.frac_violation・n_prio_viol比率・n_soft_viol比率を追加で対応ありt検定する）。

使い方:
    python -m scripts.cmp_soft_prio --base artifacts/bench_results/screen_family1_base.jsonl \\
        --var artifacts/bench_results/screen_family1_softonsoft5.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import os

from scripts.cmp_runs import _paired_t

METRICS = ("num_placed_items", "fill_score", "cog_score")
SCALE = {"num_placed_items": 100.0, "fill_score": 1.0, "cog_score": 1.0}
LABEL = {"num_placed_items": "np(pt)", "fill_score": "fill", "cog_score": "cog"}


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


def _rate(row: dict, viol_key: str, total_key: str) -> float:
    total = int(row.get(total_key, 0))
    if total <= 0:
        return 0.0
    return 100.0 * int(row.get(viol_key, 0)) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--var", action="append", default=[])
    args = parser.parse_args()

    base = _load(args.base)
    ok_base = {k: v for k, v in base.items() if v.get("status") == "success"}
    print(f"base = {os.path.basename(args.base)}  (n={len(ok_base)})")
    base_prio_rate = sum(_rate(r, "n_prio_viol", "n_prio_total") for r in ok_base.values()) / max(len(ok_base), 1)
    base_soft_rate = sum(_rate(r, "n_soft_viol", "n_soft_total") for r in ok_base.values()) / max(len(ok_base), 1)
    base_eqv = [r["equilibrium"]["frac_violation"] * 100.0 for r in ok_base.values() if "equilibrium" in r]
    print(f"   prio_viol_rate% mean {base_prio_rate:7.2f}   soft_viol_rate% mean {base_soft_rate:7.2f}"
          f"   eq_frac_viol% mean {sum(base_eqv)/max(len(base_eqv),1):7.2f}" if base_eqv else "")

    hdr = f"{'variant':<20s} {'n':>3s}"
    for m in METRICS:
        hdr += f" | {'Δ'+LABEL[m]:>9s} {'t':>6s}"
    hdr += f" | {'Δprio%':>8s} {'t':>6s} | {'Δsoft%':>8s} {'t':>6s} | {'Δeqv%':>8s} {'t':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for path in args.var:
        var = _load(path)
        common = sorted(set(ok_base) & {k for k, v in var.items() if v.get("status") == "success"})
        line = f"{os.path.basename(path)[:20]:<20s} {len(common):>3d}"
        for m in METRICS:
            d = [(float(var[k][m]) - float(ok_base[k][m])) * SCALE[m] for k in common]
            mean, t, _ = _paired_t(d)
            line += f" | {mean:>+9.2f} {t:>+6.2f}"
        d_prio = [_rate(var[k], "n_prio_viol", "n_prio_total") - _rate(ok_base[k], "n_prio_viol", "n_prio_total")
                  for k in common]
        mean_p, t_p, _ = _paired_t(d_prio)
        d_soft = [_rate(var[k], "n_soft_viol", "n_soft_total") - _rate(ok_base[k], "n_soft_viol", "n_soft_total")
                  for k in common]
        mean_s, t_s, _ = _paired_t(d_soft)
        d_eqv = [(var[k]["equilibrium"]["frac_violation"] - ok_base[k]["equilibrium"]["frac_violation"]) * 100.0
                 for k in common if "equilibrium" in var[k] and "equilibrium" in ok_base[k]]
        mean_e, t_e, _ = _paired_t(d_eqv)
        line += f" | {mean_p:>+8.2f} {t_p:>+6.2f} | {mean_s:>+8.2f} {t_s:>+6.2f} | {mean_e:>+8.2f} {t_e:>+6.2f}"
        print(line)


if __name__ == "__main__":
    main()
