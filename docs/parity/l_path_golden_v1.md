# L字経路ゴールデン生成レポート（T-016A, l_path_golden_v1）

実装詳細仕様書.md §4.5「T-016Aの最終DoD」に対応するレポート。生成日時・実行時間・
body id・一時ファイルパスは正規化対象外のため本文書には含めない。

## 公式コード同一性

- `official_git_head`（仕様参照HEAD）: `abf630f`
- `official_validator_git_blob`: `89a5e0dc5017d6d8fadd317fb9b0ffb887d9f772`
- `official_validator_file_sha256`: `bc301a9e8db8668ba1aa003e9af6e19d39c3ea79ed9b837ece61cf937974d6ae`
- `official_validator_last_commit`: `326a6baf427a3570e2686d43a81b78d104906d81`
- `generator_repo_head`（生成時リポジトリHEAD）: `225998ddd3abdf61a7646f3abc927bc6106a2ac1`
- 判定: `abf630f` の validator.py と working tree の内容は byte-identical（起動時ガードで確認済み）

## config

- `config_relpath`: `simulator/configs/sample_config.json`
- `config_file_sha256`: `8225400e80927df2e18b6d4315ae26d3202c173cdda581022c7cf3fc9ee8a6f2`
- `config_task`: `000`
- `raw_validator_config`: `{"angle_displacement_threshold": 45, "ceiling_margin": 0.018, "displacement_threshold": 0.3, "inclusion_margin": -0.005, "safety_margin": 0.015, "settle_wait_step": 300, "start_z": 0.08}`
- `effective_validator_config`（公式インスタンス属性由来）: `{"ceiling_margin": 0.018, "inclusion_margin": -0.005, "safety_margin": 0.015, "start_margin": 0.01, "start_z": 0.08}`
- `validator_config_hash`: `86583a04fd56b363f8cb6c77da56f275e7c02c81c91b82b7e81d3887022505c9`

## カテゴリ別件数・試行数・category_seed

| category | 件数 | 試行数 | category_seed |
| --- | --- | --- | --- |
| clear_path | 200 | 200 | 16924291695177937322 |
| y_leg_blocked | 200 | 200 | 13596914307113937484 |
| x_leg_blocked | 200 | 218 | 11804379198190499067 |
| safety_margin_boundary | 200 | 1775 | 4512660651835997479 |
| random_scene | 200 | 200 | 13730339348543699467 |

## safety_margin_boundary 水準別件数

| delta_mm | gap基準からの符号 | 件数 |
| --- | --- | --- |
| -20.0 | 近づける | 25 |
| -5.0 | 近づける | 25 |
| -1.0 | 近づける | 25 |
| -0.5 | 近づける | 25 |
| +0.5 | 離す | 25 |
| +1.0 | 離す | 25 |
| +5.0 | 離す | 25 |
| +20.0 | 離す | 25 |

## 全体分布・整合性

- `official_pass`: 403（>=200）
- `official_fail`: 597（>=400）
- `official_exception`: 0（==0）
- `spy_boolean_mismatch`: 0（==0）
- 総ケース数: 1000（==1000）

## 多様性診断

| category | unique_scene_input | duplicate | unique_candidate_geo | unique_blocker_geo | y_leg長 min/max | x_leg長 min/max |
| --- | --- | --- | --- | --- | --- | --- |
| clear_path | 200 | 0 | 200 | 0 | 0.202/1.259 | 0.000/0.397 |
| y_leg_blocked | 200 | 0 | 200 | 200 | 0.198/1.242 | n/a |
| x_leg_blocked | 200 | 0 | 200 | 200 | 0.186/1.208 | 0.172/0.399 |
| safety_margin_boundary | 200 | 0 | 200 | 200 | 0.240/1.191 | 0.000/0.314 |
| random_scene | 200 | 0 | 200 | 315 | 0.182/1.347 | 0.000/0.431 |

orientation分布（カテゴリ別）:

- clear_path: `{0: 41, 1: 30, 2: 35, 3: 23, 4: 34, 5: 37}`
- y_leg_blocked: `{0: 32, 1: 39, 2: 34, 3: 28, 4: 37, 5: 30}`
- x_leg_blocked: `{0: 43, 1: 23, 2: 28, 3: 35, 4: 41, 5: 30}`
- safety_margin_boundary: `{0: 26, 1: 33, 2: 41, 3: 30, 4: 25, 5: 45}`
- random_scene: `{0: 30, 1: 32, 2: 36, 3: 29, 4: 36, 5: 37}`

## 再現性

- `--verify-reproducibility` により2回生成し、正規化JSONLの一致を確認: 一致

## 正規化JSONLハッシュ

- SHA-256: `33f8da8d9dbb109586c56e5625a2a142a219e01c12ddb5cc0f23a714169003bd`
- 件数: 1000

## 生成時間

- 約 96.6 秒（`time.monotonic()` 計測）

## 出力

- `simulator/datasets/fixtures/l_path_golden_v1.jsonl`
- `docs/parity/l_path_golden_v1.md`（本ファイル）
