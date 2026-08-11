"""§2.2.1 の21点重み推定を、既知の Public Score 全件（submissions/*.txt）で検証する。

findings_2026-07.md §2.2.1 は「推定重みによる合成目的は判定に使わない」としていたが、
v46（21点フィットに含まれない真の out-of-sample 点、2026-08-05 実測 Public 60.5856）で
ΔPublic 予測が実測と誤差 0.0003 という精度を示した（findings §20）。この検証を再現可能な
形で残す。

対象は「サブスコア → Public」の写像であって「局所指標 → 本番サブスコア」ではない。
後者（局所 cog 等）は §6.1/§18.8 で3種すべて転移に失敗しており、本スクリプトが
それを保証するものではない。

使い方:
    python -m scripts.fit_public
"""
from __future__ import annotations

import glob
import json
import os
import re

# findings §2.2.1: v38 を含む21提出で解き直した実効重み。
# fill 0.369 / cog 0.264 / stab 0.177 / placement 0.000 / soft 0.191（和 1.001）。
WEIGHTS = {
    "fill_score": 0.369,
    "cog_score": 0.264,
    "stability_score": 0.177,
    "placement_score": 0.000,
    "soft_item_score": 0.191,
}

# Public Score は評価プラットフォーム側の値で submissions/*.txt には含まれないため、
# findings_2026-07.md の本文と submissions/*.txt のファイル名から手で書き起こした。
# v46/v47 は 2026-08-05 にユーザーから報告された実測値。v50/v52 は 2026-08-09〜10。
PUBLIC_SCORES = {
    "v14": 53.24, "v24": 55.00, "v25": 55.43, "v29": 59.30, "v31": 52.00,
    "v32": 59.50, "v35": 60.00, "v36": 54.00, "v37": 60.00, "v38": 61.00,
    "v39": 57.60, "v40": 60.2, "v41": 57.4, "v42": 57.4, "v43": 61.5,
    "v44": 61.3, "v45": 61.4, "v46": 60.5856, "v47": 54.48, "v50": 63.14,
    "v52": 63.16,
}

# 隣接する診断提出の Δ 予測精度（意思決定に実際に使われた遷移）。
TRANSITIONS = [
    ("v14", "v24"), ("v24", "v25"), ("v25", "v29"), ("v29", "v31"), ("v29", "v32"),
    ("v32", "v35"), ("v35", "v36"), ("v35", "v38"), ("v38", "v39"), ("v38", "v40"),
    ("v38", "v41"), ("v38", "v42"), ("v38", "v43"), ("v43", "v44"), ("v43", "v45"),
    ("v43", "v46"), ("v43", "v47"), ("v46", "v47"), ("v46", "v50"), ("v43", "v50"),
    ("v50", "v52"),
]

_TAG_RE = re.compile(r"submit_\d{8}_(v\d+)(?:_|\.)")


def _load_submissions(submissions_dir: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(submissions_dir, "submit_*.txt"))):
        m = _TAG_RE.match(os.path.basename(path))
        if not m:
            continue
        tag = m.group(1)
        if tag not in PUBLIC_SCORES:
            continue
        with open(path) as f:
            rows[tag] = json.load(f)
    return rows


def _predict(row: dict) -> float:
    return sum(WEIGHTS[k] * float(row[k]) for k in WEIGHTS)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    submissions_dir = os.path.join(here, "..", "submissions")
    rows = _load_submissions(submissions_dir)
    missing = sorted(set(PUBLIC_SCORES) - set(rows))
    if missing:
        print(f"warning: submissions/*.txt に見つからないタグ: {missing}")

    tags = sorted(rows, key=lambda t: int(t[1:]))
    preds_raw = {t: _predict(rows[t]) for t in tags}
    actual = {t: PUBLIC_SCORES[t] for t in tags}

    intercept_all = sum(actual[t] - preds_raw[t] for t in tags) / len(tags)
    tags_no_v31 = [t for t in tags if t != "v31"]
    intercept_ex = (sum(actual[t] - preds_raw[t] for t in tags_no_v31) / len(tags_no_v31)
                     if tags_no_v31 else intercept_all)

    print(f"weights: {WEIGHTS}  (sum={sum(WEIGHTS.values()):.3f})")
    print(f"n={len(tags)}  intercept(all)={intercept_all:+.3f}  "
          f"intercept(excl v31)={intercept_ex:+.3f}\n")

    print(f"{'tag':<5s} {'actual':>8s} {'pred':>8s} {'resid':>8s}")
    resid_all, resid_ex = [], []
    for t in tags:
        pred = preds_raw[t] + intercept_ex
        r = actual[t] - pred
        resid_all.append(r)
        if t != "v31":
            resid_ex.append(r)
        print(f"{t:<5s} {actual[t]:8.3f} {pred:8.3f} {r:+8.3f}")

    def _rms(xs: list[float]) -> float:
        return (sum(x * x for x in xs) / len(xs)) ** 0.5 if xs else float("nan")

    print(f"\nresid RMS (all, n={len(resid_all)})       = {_rms(resid_all):.3f}"
          f"  max|resid| = {max((abs(r) for r in resid_all), default=float('nan')):.3f}")
    print(f"resid RMS (excl v31, n={len(resid_ex)}) = {_rms(resid_ex):.3f}"
          f"  max|resid| = {max((abs(r) for r in resid_ex), default=float('nan')):.3f}")

    print("\ndelta-prediction on submitted transitions (uses raw weights, no intercept):")
    print(f"{'transition':<12s} {'actual':>8s} {'pred':>8s} {'err':>8s}")
    for a, b in TRANSITIONS:
        if a not in rows or b not in rows:
            continue
        dp = preds_raw[b] - preds_raw[a]
        da = actual[b] - actual[a]
        print(f"{a}->{b:<7s} {da:+8.2f} {dp:+8.2f} {da - dp:+8.2f}")


if __name__ == "__main__":
    main()
