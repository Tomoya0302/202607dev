"""方策3(HF-030)判定用ベンチタスク生成器（Phase 0、findings_2026-07.md §13）。

`configs/local_suite/c01.json`（1容器）/`c03.json`（2容器）の実コンテナ・camera・validator・
action・visualizer・agent 設定をテンプレートとして再利用し、`item_stream.item_list` のみを
差し替えた課題A（`optimize=True, look_ahead=1`）タスクを大量生成する。

7SKU閉集合はローカルの実課題7件（c01-c05, sample_config.json の2課題）全項目から実測で
復元した（`docs/findings_2026-07.md` §3 の「7 SKU 閉集合」の裏取り）。各SKUは長さ/幅/高さ/
質量/摩擦/反発係数/is_soft が固定（実データで確認済み）。is_prioritized のみ課題ごとに
独立抽選する（実データでの観測比率 約9%、26/288品目）。

真の課題生成規則（混合比・個数分布）は非公開（findings §3/§7.0）なので、本生成器は
「既知の実課題の集合をそのまま拡張する」という最小限の仮定のみを置く：SKU出現頻度は
実測値（7課題・288品目）に比例、個数は家族の代表値（1容器41品目=c01実測、2容器80品目=
findings §3「80個=2台」引用）に固定。CLAUDE.md「不明な仕様は推測で大きく変更しない」に
従い、コンテナ形状・物理パラメータは実課題からの改変なしコピーとする。
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np

# --- 7SKU閉集合（実データ復元、docs/findings_2026-07.md §3 参照） -----------------------
# (length, width, height, mass, is_soft, lateralFriction, rollingFriction, spinningFriction, restitution)
_SKU_TABLE = [
    (0.45, 0.30, 0.20, 5.0, True, 0.8, 0.02, 0.02, 0.0),
    (0.50, 0.40, 0.40, 10.0, True, 0.6, 0.01, 0.01, 0.1),
    (0.55, 0.40, 0.24, 8.0, False, 0.4, 0.01, 0.01, 0.2),
    (0.60, 0.30, 0.25, 7.0, True, 0.8, 0.02, 0.02, 0.0),
    (0.65, 0.35, 0.23, 12.0, True, 0.8, 0.02, 0.02, 0.0),
    (0.65, 0.45, 0.25, 13.0, False, 0.4, 0.01, 0.01, 0.2),
    (0.75, 0.56, 0.27, 18.0, False, 0.4, 0.01, 0.01, 0.2),
]
# 実測出現頻度（7課題・288品目の合計カウント、docs/findings_2026-07.md §3 の裏取りで実測）。
_SKU_WEIGHTS = np.array([13, 14, 93, 25, 37, 80, 26], dtype=np.float64)
_SKU_WEIGHTS = _SKU_WEIGHTS / _SKU_WEIGHTS.sum()
_PRIO_RATE = 26.0 / 288.0  # 実測 is_prioritized 比率（6課題×4 + 1課題×2 / 288品目）

_FAMILY_TEMPLATES = {
    1: {"template": "configs/local_suite/c01.json", "n_items": 41},
    2: {"template": "configs/local_suite/c03.json", "n_items": 80},
}


def _load_template(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    return d[next(iter(d.keys()))]


def _gen_item_list(rng: np.random.Generator, n_items: int) -> list[dict]:
    sku_idx = rng.choice(len(_SKU_TABLE), size=n_items, p=_SKU_WEIGHTS)
    prio = rng.random(n_items) < _PRIO_RATE
    items = []
    for i in range(n_items):
        length, width, height, mass, is_soft, lf, rf, sf, rest = _SKU_TABLE[sku_idx[i]]
        items.append({
            "index": i,
            "length": length, "width": width, "height": height, "mass": mass,
            "is_prioritized": bool(prio[i]), "is_soft": is_soft,
            "lateralFriction": lf, "rollingFriction": rf, "spinningFriction": sf,
            "restitution": rest,
        })
    return items


def gen_task(family: int, seed: int) -> dict:
    """family(1|2) と seed から課題A タスク1件（`{"000": {...}}` 形式）を生成する。"""
    spec = _FAMILY_TEMPLATES[family]
    template = _load_template(spec["template"])
    rng = np.random.default_rng(seed)
    inner = copy.deepcopy(template)
    inner["item_stream"]["item_list"] = _gen_item_list(rng, spec["n_items"])
    inner["item_stream"]["look_ahead"] = 1
    inner["item_stream"]["max_space"] = 1
    inner["item_stream"]["visible_pool"] = []
    inner["agent"]["optimize"] = True
    for c in inner["containers"]["container_list"]:
        c["packed_items"] = []
    return {"000": inner}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=int, choices=(1, 2), required=True)
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=20260801)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n_tasks):
        seed = args.seed_base + i
        task = gen_task(args.family, seed)
        out_path = os.path.join(args.out_dir, f"f{args.family}_{i:04d}.json")
        with open(out_path, "w") as f:
            json.dump(task, f)
    print(f"generated {args.n_tasks} tasks for family {args.family} -> {args.out_dir}")


if __name__ == "__main__":
    main()
