"""`stair_mass_batch.py` の出力を集計し、質量と配置高さ(z_norm)の相関を条件間で比較する
（findings §21.3の仮説検証: 階段制約の設計＝STAIR_SLACK絶対値 vs STAIR_K比例で、
重い荷物ほど高く置かれる傾向が変わるか）。

使い方:
    python -m scripts.stair_mass_report --jsonl a.jsonl --label baseline \
        --jsonl b.jsonl --label stair_slack --jsonl c.jsonl --label stair_k
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _load_items(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "error" in row:
                continue
            items.extend(row.get("items", []))
    return items


def _weighted_slope(mass: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """単純OLS（mass -> z_norm）の傾きと標準誤差を返す。"""
    n = len(mass)
    if n < 3:
        return 0.0, 0.0
    x = mass - mass.mean()
    y = z - z.mean()
    sxx = float(np.sum(x * x))
    if sxx <= 1e-12:
        return 0.0, 0.0
    slope = float(np.sum(x * y) / sxx)
    resid = y - slope * x
    dof = max(n - 2, 1)
    sigma2 = float(np.sum(resid ** 2)) / dof
    se = float(np.sqrt(sigma2 / sxx))
    return slope, se


def summarize(label: str, path: str) -> None:
    items = _load_items(path)
    n = len(items)
    print(f"\n=== {label} ({path}): n_items={n} ===")
    if n < 3:
        print("  (too few items)")
        return
    mass = np.array([float(it["mass"]) for it in items])
    z = np.array([float(it["z_norm"]) for it in items])
    is_soft = np.array([bool(it.get("is_soft")) for it in items])

    corr = float(np.corrcoef(mass, z)[0, 1]) if np.std(mass) > 0 and np.std(z) > 0 else 0.0
    slope, se = _weighted_slope(mass, z)
    t = slope / se if se > 1e-12 else 0.0
    print(f"  corr(mass, z_norm) = {corr:+.4f}")
    print(f"  slope dz_norm/dmass = {slope:+.5f} (se={se:.5f}, t={t:+.2f})  "
          f"[意味: 質量+1kgあたり相対高さが何変わるか。正=重いほど高く置かれる]")

    # 質量を四分位に分けて平均 z_norm を出す（直感的な要約）
    order = np.argsort(mass)
    q = max(n // 4, 1)
    light_z = z[order[:q]].mean()
    heavy_z = z[order[-q:]].mean()
    print(f"  mean z_norm: 軽量25%={light_z:.4f}  重量25%={heavy_z:.4f}  差={heavy_z-light_z:+.4f}")

    for kind, mask in (("soft", is_soft), ("hard", ~is_soft)):
        if mask.sum() >= 3:
            c = float(np.corrcoef(mass[mask], z[mask])[0, 1]) if np.std(mass[mask]) > 0 else 0.0
            print(f"  [{kind:4s}] n={int(mask.sum())}  corr(mass,z_norm)={c:+.4f}  "
                  f"mean_z={z[mask].mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    args = parser.parse_args()
    if len(args.jsonl) != len(args.label):
        raise SystemExit("--jsonl と --label の数は一致させてください")
    for path, label in zip(args.jsonl, args.label):
        summarize(label, path)


if __name__ == "__main__":
    main()
