# interface_notes.md — 公式実装読解メモ（T-002 成果物）

| 項目 | 内容 |
| --- | --- |
| 目的 | `src/ground_handling/`（公式配布コード）を一次情報として読解し、`packing_core/constants.py` へ転記する値の根拠を記録する |
| 位置づけ | 実装詳細仕様書 §3.5/§3.6、手順書 §1.5 の仕様仮定台帳と対をなす一次資料 |
| 読解方針 | 公式コードの**変更・コピーは禁止**（L字経路移植 T-016 のみ例外、本メモでは未実施）。読解のみ |
| 更新ルール | 公式実装との不一致が見つかった場合、本メモを修正してから `constants.py` を直す |

読解対象: `validator.py` `evaluator.py` `containers.py` `items.py` `env.py` `runner.py` `camera.py` `utils.py`、
`configs/sample_config.json`（task 000/001）、`configs/item_params.xlsx`、`simulator/README.md`。

---

## A. 公式検証パラメータ（評価基盤・確定）

出典を **2 系統**（README 明記値／実配布 `sample_config.json` の task 000・001）で二重確認済み。両者完全一致。

| パラメータ | 値 | 単位 | 根拠 |
| --- | --- | --- | --- |
| `inclusion_margin` | **-0.005** | m | README:199 / sample_config.json（両task）/ validator.py:78, evaluator.py:56 |
| `safety_margin` | **0.015** | m | README:200 / config / validator.py:17,236 |
| `ceiling_margin` | **0.018** | m | README:201 / config / validator.py:21,130 |
| `displacement_threshold` | **0.3** | m | README:202 / config / validator.py:182,203 |
| `angle_displacement_threshold` | **45** | deg | README:203 / config / validator.py:183,203 |
| `start_z` | **0.08** | m | README:204 / config / validator.py:20,115 |
| `settle_wait_step` | **300** | step | README:205 / config / validator.py:189 |
| `start_margin`（README未記載・既定値） | 0.01 | m | validator.py:18 `config.get('start_margin', 0.01)`（sample_configにキーなし→既定採用）|

参考: エージェント実行タイムアウトは README:147 と sample_config.json task000 `agent` で一致
（`init_timeout=10.0` / `optimization_timeout=180.0` / `policy_timeout=8.0` / `max_mem=12`）。

---

## B. L字搬入経路（`validator.py::check_transport_path`, L85–175）

- **移動順**: ①入口 `y = -width/2`（開口面、-Y側）から**+Y へ前進**して target の Y へ（x, z 固定, L146–150）
  → ②**X 方向へ横移動**して target の X へ（y, z 固定, L160–164）。＝「Y前進 → X横移動」の L 字経路。
- **入口レーン x**（仮定 A12 関連）: `x = clip(target_x, x_min, x_max)`（L99）
  - `x_min = -length/2 + thickness + cut_x + half_lwh[0] + start_margin`（L96）
  - `x_max =  length/2 - thickness - half_lwh[0] - start_margin`（L97）
- **搬送高さ**: `z = target_z + effective_start_z`（上端/天井でクランプ, L135）。既定 `effective_start_z = start_z = 0.08`。
- **`start_z` が 0 になる条件**（直置き, L118–123）: 荷物底面 `bottom_z = target_z - half_h` が
  直置き面 `resting_surfaces = [thickness, height/2 + thickness + buffer]`（床面／棚上面）の
  **直上 0〜0.05m 以内**にあるとき、`effective_start_z = 0`。
- **天井クランプ**（L126–133）: 天井面 `ceiling_surfaces = [height/2+buffer, height+buffer-thickness]`
  （棚下面／コンテナ天井）との隙間 `clearance` が `effective_start_z + ceiling_margin` 未満のとき、
  `effective_start_z = max(0, clearance - ceiling_margin - 0.0005)` にクリップ。
- 干渉判定は `getClosestPoints(distance=safety_margin)` を他の既配置荷物・棚・小棚それぞれに対して実施（L230–274）。
  1件でも検出されれば `is_packable = False`。

---

## C. 包含判定と符号解釈（仮定 A14 の核心・**要修正あり→§I 参照**）

- `validator.py::check_inclusion`（L77–78）:
  ```
  dots = Σ n_vec · (target_center - point) + |n_vec| · half_lwh
  合格条件: all(dots <= inclusion_margin)
  ```
  `n_vec` は**内壁面から外側へ向かう単位法線ベクトル**（README:162、utils.py:200–213 で CCW多角形オフセットから導出）。
- **符号の意味**（`dots` の定義から直接導出。数値シミュレーションでも再確認済み）:
  - `dots` は「荷物のその面が壁からどれだけ外側にあるか」（壁位置=0、外側が正）。
  - `margin` が **正** → `dots<=margin` は正の超過まで許すので**はみ出しを許容（緩和）**。
  - `margin` が **負** → 荷物面が壁の**内側**にどれだけ入っていないといけないかを要求（**厳格**）。
  - `inclusion_margin = -0.005` の場合: 壁にちょうど接触する荷物（`dots=0`）は**不合格**。
    内側へ **5mm 以上**引っ込んでいないと合格しない。
- `evaluator.py::calculate_fill_rate`（L48–64）: 回転後の8隅それぞれで
  `n_vec·(corner-point) > inclusion_margin` が1つでも成立すれば「コンテナ外」と判定。
  **完全に内包された荷物のみ**体積を fill の分子に加算（`out_items` に理由付きで記録）。
- `env.py:51` は `Evaluator` にも **`validator`と同一の** `inclusion_margin` を渡している
  （`config={'inclusion_margin': self.config['validator']['inclusion_margin']}`）。
  → README:377「内包判定は検証時よりも緩く設定されている」との**記述との不一致**（§I-5 参照）。

---

## D. 評価スコア（README §評価指標:373–382 + フィードバックJSON例:461–486）

| スコア | 定義（確定度） | 出典 |
| --- | --- | --- |
| `fill_score` | **確定**: `min(100 · Σ完全内包item体積 / Σ有効体積, 100)`（evaluator.py:70）。分母は `Σ container.volume` | evaluator.py |
| `cog_score` | 荷物全体の重心が低いほど高スコア（**式・重み非開示**。定性説明のみ） | README:378 |
| `stability_score` | 動的揺らし試験での不動性・非崩壊性（**式・重み非開示**） | README:380 |
| `placement_score` | 優先手荷物が他属性の下敷き（上方接触）や非優先コンテナへの配置で減点（**式・重み非開示**） | README:379 |
| `soft_item_score` | ソフト貨物が非ソフトの下敷きで減点、優先/ソフトは独立評価（**式・重み非開示**） | README:379 |
| `num_placed_items` | `Σ packed_items / total_items`（evaluator.py:78） | evaluator.py |
| 「一定数以上」閾値 | 「一定数以上積めないと fill 以外 0」との記述はあるが**具体数値は非開示**。`evaluator.py` にも該当ロジックなし | README:382 |

配布 `evaluator.py::evaluate()`（L75–86）が実際に返すキーは `fill_score` と `num_placed_items` の**2つのみ**。
`cog_score` / `stability_score` / `placement_score` / `soft_item_score` の計算式・重みは**配布コードに実装がない**
（評価基盤側のみに存在すると推測される）。→ 仕様仮定台帳 A5「配布evaluator.py の式・重み＝評価基盤の式・重み」は
本読解の結果、**確認不能のため unconfirmed を維持**（§I-2）。

> 有効体積 `container.volume`（containers.py:101–114）:
> `base − cut − small_shelf − shelf`
> - `base = (L-2t)(W-2t)(H-t-buffer)`
> - `cut = 0.5·(cut_x-t)·(cut_y-t)·(W-2t)`
> - `small_shelf = cut_x · t · (W-2t)`
> - `shelf = (L-2t)·t·(W/2-2t)`（`require_shelf=True` のときのみ加算）
> （`t=thickness`, `L/W/H`はコンテナ外寸, `buffer`は天井バッファ）

---

## E. observation / init_states キー転記表

| 区分 | キー | 根拠 |
| --- | --- | --- |
| `init_states`（`get_init_states`引数） | `optimize`, `lookahead_k`, `container_list` | env.py:170–176 |
| `observation`（`policy`へ渡る最終形、shm復元後） | `optimize`, `lookahead_k`, `depth_map`, `container_list`, `pool_list` | env.py:278–286 + runner.py:38–57 |
| 復元前の生 observation（shm経由） | 上記のうち `depth_map` の代わりに `shm_name` / `shm_shape` / `shm_dtype` | env.py:279–281 |
| `container_list` 要素 | `index, length, width, height, cut_x, cut_y, thickness, center, n_vecs, points, volume, shelf, is_prioritized, packed_items` | containers.py:314–336 |
| `item_list` / `pool_list` 要素（共通） | `index, length, width, height, mass, is_prioritized, is_soft, belongs_to, pos, orn, lateralFriction, rollingFriction, spinningFriction, restitution, angularDamping` | items.py:42–68 |
| item のソフト限定追加キー | `contactStiffness, contactDamping, linearDamping`（`is_soft=True` のときのみ付与） | items.py:61–66 |
| `action`（`policy`戻り値） | `item_idx, container_idx, place_pos, orientation` | validator.py:30 / README:295–302 |

**注意点**:
- 公式コードのキーは **`n_vecs`（複数形）**（containers.py:325）。README:254 の記述は `n_vec`（単数）で**ドキュメント誤記**。
  → `constants.OBS_KEYS` の転記は公式コード優先で `n_vecs` を採用。
- `belongs_to` / `pos` / `orn` は未配置時 `None`（items.py:16–19）。

---

## F. Env / Runner フロー・プール管理

- **プール管理**（`ItemStreamManager`, items.py:162–241）:
  - `all_items`: 全荷物（初期順、`set_order`で並べ替え可）。`visible_pool`: 現在選択可能なプール。
  - `lookahead_k = config['look_ahead']`。`max_space = config['max_space']`（補充しきい値）。
  - `get_item(pool_index)` は `visible_pool[pool_index]` を返す（→ `item_idx` は**プール内 index**、仮定A11）。
  - `pop_and_refill(pool_index)`（L225–236）: 該当要素を pop → 空き数が `max_space` 以上なら空き分を補充。
  - `set_order(order)`（L183–185）: `all_items` を `order`（index列）で並べ替え。`reset()` はプールを空にしてから
    `lookahead_k` 個を補充。
- **optimize 分岐**: `init_states['optimize']`（env.py:173, 元は `config['agent']['optimize']`）。
- **optimize→policy 呼び出し順（env.py 公開APIとREADME:233–235 からの推定。厳密な駆動コードは
  `app.py`/`scripts/run_test.py` にあり、今回の読解対象外）**:
  1. `agent = Agent(module_path)`
  2. `agent.get_init_states(init_states)`（`init_states = env.get_init_states()`）
  3. `optimize=True` の場合: `item_list = env.get_info_for_optimization()`（=全item, env.py:179）→
     `order = agent.optimize(item_list)` → `env.set_item_order(order)`（L183–187、
     `set(order) == stream_manager.all_indices` を検証、不一致なら無視されFalse）
  4. `env.reset()` でプール充填 → 毎ステップ `observation = env._get_obs()` →
     `action = agent.policy(observation)` → `env.step(action)`
  - 上記手順4〜5はREADMEの記述と`env.py`のメソッド境界からの論理的推定であり、`app.py`側の実際の呼び出しコードは
    読解対象外のため**未検証**。必要なら人間側で `app.py` / `scripts/run_test.py` を確認されたい。

---

## G. Camera（`camera.py`）

- `depth_map` の shape は `(コンテナ数, img_height, img_width)`。評価基盤解像度は **64×64**
  （env.py:81–82, README:217–218, camera.py:122）。
- カメラは**開口面（-Y側）に固定**: `eye=[center_x, door_y-distance, center_z]`, `target=center`,
  `up=[0,0,1]`（camera.py:64–72）。→ 視線方向は **+Y**。
- **軸対応（読解結果）**: 画像横方向 `u` → 世界 **X**、画像縦方向 `v` → 世界 **Z**、
  **画素値**（`available_depth`, L125–130）→ 開口面から荷物までの **Y 距離**。
  - ⚠ **README:315–321 の変換式と矛盾**: README は `local_y = f(u)`, `... `のような対応で
    `v → local_y` の関係を示唆しているが、camera.py の view/projection 行列読解では `v → Z`, `値 → Y`
    となる。**どちらが実際の挙動か、GUI実測 (`--render-mode human`) での確認が必要**
    （手順書 Phase0 L177 が指示する「実測確認」項目）。本メモでは **unconfirmed** とし、
    `constants.py` には軸対応の断定値を転記しない。
- heightmap 復元について: `depth_map` は正面（開口面）からの直交投影デプスであり、真上からの
  heightmap ではない。内部状態（EMS等）の再構築は `packed_items`（沈降後の実姿勢・回転後AABB）から
  行う方針（手順書 Phase1:192）。`depth_map` は観測特徴の補助として扱う。

---

## H. Item 種別表（`configs/item_params.xlsx` → `KIND_IS_SOFT` / `POOL_MAX_WEIGHT`）

| 種別（xlsx列/和名） | is_soft (row7) | mass[kg] (row5) | L/W/H [m] (rows 2-4) |
| --- | --- | --- | --- |
| スーツケース(大) | False (0) | 18 | 0.75 / 0.56 / 0.27 |
| スーツケース(中) | False (0) | 13 | 0.65 / 0.45 / 0.25 |
| スーツケース(小) | False (0) | 8 | 0.55 / 0.40 / 0.24 |
| ダッフル/ボストン | True (1) | 7 | 0.60 / 0.30 / 0.25 |
| 段ボール | True (1) | 10 | 0.50 / 0.40 / 0.40 |
| 大型リュックサック | True (1) | 12 | 0.65 / 0.35 / 0.23 |
| 小型デイパック | True (1) | 5 | 0.45 / 0.30 / 0.20 |

- **`POOL_MAX_WEIGHT = 18.0`**（xlsx 最大 mass = スーツケース(大)。`sample_config.json` の item_list も
  mass レンジ `[5, 18]` で一致）。
- ランタイムの item 辞書（§E参照）に `kind` フィールドは**存在しない**。`is_soft` は各荷物に直接付与される。
  そのため `KIND_IS_SOFT` は主に学習データ合成・参照用の表であり、評価基盤の実データ判定には使わない。
- README:456「評価基盤ではコンテナ数・寸法・荷物の種類・数・順序が sample_config と全く異なる」との明記あり。
  → `POOL_MAX_WEIGHT` は正規化用の**参照上限**として使い、実eval値がこれを超える可能性を考慮して
  下流では `[0, 1]` へのクランプを推奨する。

---

## I. 仕様書と公式実装の不一致（要報告事項）

1. **【重要】A14 符号解釈の誤り**: 詳細仕様書 §3.6（旧記述）「`inclusion_margin=-0.005` は
   『5mm のはみ出しまで許容』の意」は**符号の向きが逆**。正しくは「`-0.005` は内側 5mm のクリアランスを
   要求する厳格側」（§C 参照）。→ 本チケットで §3.6 の文言を修正し、`constants.ASSUMPTIONS["A14"]` を
   confirmed 化する（別途 `constants.py` 変更で反映）。
2. **`evaluator.py` の不完全性（仮定 A5 関連）**: 配布 `evaluator.py` は `fill_score` と
   `num_placed_items` のみを計算する。`cog_score` / `stability_score` / `placement_score` /
   `soft_item_score` の式・重み、および「一定数以上」の具体的閾値は**配布コードに存在しない**。
   → `constants.RegimeParams.regime_threshold_n` は本チケットでは確定できず、`-1`（未確定）を維持し、
   運営への質問リストに追加する。仕様仮定台帳 A5 は unconfirmed を維持。
3. **README の `n_vec` 表記ゆれ**: README:254 は `n_vec`（単数形）と記載するが、公式コードの実キーは
   `n_vecs`（複数形、containers.py:325）。転記は実装優先で `n_vecs` を採用。
4. **Camera 軸対応の矛盾**: README:315–321 の変換式と camera.py の view/projection 読解が食い違う
   （§G 参照）。GUI実測での確認が必要。`constants.py` には未転記。
5. **fill 内包マージンの「緩さ」記述との不一致**: README:377 は「内包判定は検証時よりも緩く設定されている」
   と記述するが、配布コード上は `env.py:51` で validator と evaluator が**同一の** `inclusion_margin`
   を共有している（§C 参照）。評価基盤側で別値を使っている可能性があるため、運営照会に含める。
6. **課題対応の数値ズレ（軽微・実害なし）**: 手順書1.5「課題B=lookahead≈20」との記述に対し、
   sample_config.json task001 は `look_ahead=10`。もともと仮定A1は未確認(Q1)であり、
   `sample_config.json` はローカル動作確認用で評価基盤とは別分布（README:456）と明記されているため
   台帳上は不整合ではない。ハードコード禁止方針のため実装への影響なし。

---

## J. 付録D形式：運営への質問リスト（候補）

```
[質問] regime_threshold_n の根拠
1. 停止した箇所: RegimeParams.regime_threshold_n（constants.py）
2. 矛盾・不明の内容: README「手荷物を一定数以上コンテナに積載できていないと充填率スコア以外は0」との
   記述に対し、配布 evaluator.py にはこの閾値のロジック・数値が実装されていない。
3. 候補解釈: (a) 評価基盤側にのみ実装され非開示 (b) 「一定数」=1個（1個も置けない場合のみ0）
   (c) 全体の割合（例: 50%）で判定
4. 推奨: 数値の開示、または少なくとも定義（絶対数 or 割合）の開示を要望
```

```
[質問] cog/stability/placement/soft_item スコアの式・重み
1. 停止した箇所: 評価スコア全体（evaluator.py には fill/num_placed のみ実装）
2. 矛盾・不明の内容: README は定性的な説明のみで、計算式・重みが非開示。仕様仮定台帳 A5
   「配布evaluator.py の式・重み＝評価基盤の式・重み」が配布コード側に実体がなく検証不能。
3. 候補解釈: (a) 非開示のまま提出者側で近似実装し、LBフィードバックで較正する方針を継続
   (b) 追加資料で開示される予定がある
4. 推奨: 開示が難しい場合、(a)の方針で進める旨を確認したい
```

```
[質問] fill_score の内包マージンと Camera の軸対応
1. 停止した箇所: evaluator.py の inclusion_margin 使用箇所 / camera.py の depth_map 軸対応
2. 矛盾・不明の内容:
   - README:377「内包判定は検証時よりも緩く設定されている」に対し、配布コード(env.py:51)は
     validator と evaluator が同一の inclusion_margin を共有している。
   - README:315-321 の depth_map→(x,y)変換式が camera.py の視点設定（開口面から+Y視線）の
     読解結果（v→Z, 画素値→Y）と整合しない。
3. 候補解釈:
   - マージン: (a) 評価基盤側では別値を使用 (b) 配布コードのまま
   - Camera: (a) README記述が正 (b) 読解(v→Z,値→Y)が正 (c) GUI実測で確認要
4. 推奨: 評価基盤の実際の値・軸対応の確認、または実測での確認方針の是非
```

---

## 更新履歴

| 日付 | 内容 |
| --- | --- |
| 2026-07-17 | 初版（T-002）。§A〜H 転記表、§I 不一致一覧、§J 質問リスト |
