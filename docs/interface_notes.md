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
- **`spacing` は `init_states`/`observation` のいずれにも存在しない**（上表の転記済みキー一覧参照）。
  `config['spacing']` は `MultiContainerManager.build`（containers.py:270,273）が内部でのみ読む値であり、
  agent 側には渡らない。コンテナ原点世界X（`offset_x`）が必要な場合は `cdict["center"][0]` を
  **直接**使用し、`spacing` からの逆算・`index * spacing` の再計算は行わない（詳細・根拠は §I-8）。

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
7. **【T-006で発覚・解決済】`container_space.py` 仕様（旧§4.2）の `buffer` キー前提の誤り**:
   旧仕様書は `cdict`（container_list要素）に `buffer` キーが存在する前提で
   `inner_min_rel`/`inner_max_rel` を `thickness`/`buffer` の寸法式から再構築する設計だったが、
   `buffer` は `containers.py:24` の `Container.buffer`（既定0.01、**configで上書き可能**）であり、
   `get_item_info_in_containers()`（containers.py:314–336）が生成する dict には**転記されていない**
   （§E の転記表・`constants.OBS_KEYS["container"]` にも `buffer` は含まれない）。
   さらに `sample_config.json` では `buffer=0.0` に上書きされており、固定値 0.01 で補うことも不正確。
   → **解決方針**（人間承認済み・2026-07-17）：`CONTAINER_BUFFER` 等の固定値は `constants.py` に追加しない。
   `container_space.py` は `cdict["points"]`（世界座標の代表点）と `cdict["n_vecs"]`（外向き法線）を
   正として内壁形状を復元する（半空間 `normal_rel·x <= d` の交差、§4.2 参照）。
   `buffer` がどうしても必要な場合のみ、公式実装の関係式
   `buffer = float(cdict["center"][2]) - float(cdict["height"]) / 2.0` から都度復元し、キー欠落を
   固定値で埋めない。`実装詳細仕様書.md §4.2` を本方針で修正済み。
8. **【T-012計画中に発覚・解決済】`state.py`/`container_space.py` 仕様（旧§3.1/§4.2）の
   `offset_x = index * spacing` 前提の誤り**:
   旧仕様書 §3.1 は「コンテナ i の原点世界X = `offset_x_i = i * spacing`（`spacing` は
   `init_states` から取得）」としていたが、`spacing` は agent I/F（`init_states`: env.py:170–176 /
   `observation`: env.py:278–286）のいずれにも**存在しない**（§E 参照）。`config['spacing']` は
   `MultiContainerManager.build`（containers.py:270,273）が `offset_x = i * spacing` の算出に
   内部で使うのみで、agent 側には転記されない。
   一方、コンテナ原点世界Xは `cdict["center"][0]` として観測可能である。公式コード
   `containers.py:61` `pos=(0,0,hz+buffer)`（局所座標、x=0）→ `containers.py:66`
   `self.center = self.local_to_global(pos)` → `containers.py:238–240`
   `local_to_global(l) = (l[0]+offset_x, l[1], l[2])` により、**`center[0] == offset_x`** が
   恒等的に成り立つ（`containers.py:243–245` の `global_to_local(g) = (g[0]-offset_x, g[1], g[2])` も
   §3.1 の X 限定オフセット変換と一致）。
   → **解決方針**（人間承認済み・2026-07-17）：`offset_x = float(np.asarray(cdict["center"])[0])` を
   **直接**使用する。`center.x` から `spacing` を逆算したり `index * spacing` を再計算したりしない
   （複数コンテナの座標変換不具合を隠す可能性があるため）。`build_container_space` の signature から
   `spacing` 引数を削除する。`実装詳細仕様書.md §3.1/§4.2/§4.4` および `初期検討_実装手順書.md §1.2` を
   本方針で修正済み。なお `simulator/` 側の実装・テスト（`container_space.py` 等）の追随は別タスクとし、
   本訂正時点では未着手（既存 golden fixture は `center[0]==index*spacing` で自己無矛盾のため、
   実装移行後も期待値の数値は変わらない見込み）。

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

## K. T-007 cut・棚の変換規則（読解結果、一次情報: `ground_handling/containers.py` / `utils.py`）

読解専用（コピー禁止）。以下は `container_space.py`（T-007）が採用する変換規則の根拠記録。

### K.1 `buffer` の復元（§I-7 の再確認）

`buffer = float(cdict["center"][2]) - float(cdict["height"]) / 2.0`
（`containers.py:61,66`: `pos=(0,0,hz+buffer)` → `center=local_to_global(pos)` より逆算）。
棚AABBの構築にも同じ復元式を用いる。

### K.2 小棚（small shelf）

出典: `containers.py::_create_small_shelf`（L194–213）。呼び出し元 `create()` の
L88–91（`require_shelf=True`側）と L94–98（`require_shelf=False`側）は**同一パラメータで
両分岐とも呼び出す**。したがって小棚の存在は `cdict["shelf"]` の真偽に依存しない
（公式コード上は常時生成）。

半空間ではなく PyBullet の `halfExtents` ボックスとして定義される。コンテナ本体と同一の回転
（Euler(π/2,0,0)）を適用し、`inner_min_rel`/`inner_max_rel` と同じコンテナ相対座標系に変換すると：

```python
center_rel = (-length/2 + cut_x/2 + thickness, 0.0, height/2 + thickness/2 + buffer)
half        = (cut_x/2, width/2 - thickness, thickness/2)
raw_min = center_rel - half
raw_max = center_rel + half
```

**生成条件**: `cdict["shelf"]` の真偽・`cut_planes` の有無にかかわらず、常に上記AABBを計算する
（→ K.4 のクリップ・正体積フィルタを経て採否を決める）。

### K.3 大棚（main shelf）

出典: `containers.py::_create_shelf`（L216–235）。呼び出し元 `create()` L81–86、
`require_shelf=True` のときのみ呼ばれる。

```python
center_rel = (0.0, width/4, height/2 + thickness/2 + buffer)
half        = (length/2 - thickness/2, width/4 - thickness, thickness/2)
raw_min = center_rel - half
raw_max = center_rel + half
```

**生成条件**: `cdict["shelf"] is True` のときのみ上記AABBを計算する。`cut_planes` の有無には
依存しない。

### K.4 内壁AABBへのクリップと正体積フィルタ

数値確認: 大棚のX半径 `length/2 - thickness/2` は、内壁AABBのX境界
（軸整列面から復元される `length/2 - thickness` 相当）より**壁厚の半分（`thickness/2`）だけ外側**
に出る（棚が壁へ構造的に食い込む設計のため）。そのままでは有効空間の外に障害物AABBがはみ出すため、
次のクリップを適用してから採用する：

```python
clip_min = elementwise_max(raw_min, inner_min_rel)
clip_max = elementwise_min(raw_max, inner_max_rel)
```

さらに、クリップ後の各軸長 `clip_max[k] - clip_min[k]` が**全軸で `EPS_GEOM` より大きい場合のみ**
`shelf_boxes` に追加する（ゼロ・負の寸法になる場合は追加しない）。この正体積フィルタは、退化した／
ゼロ寸法の合成fixtureを安全に扱うための一般的な幾何正規化であり、特定fixtureのための特別分岐ではない。

### K.5 T-006 合成fixtureの既知の不整合（本番コードでは救済しない）

既存 `tests/test_container_space.py::_box_cdict` は「cutなし直方体」を意図しつつ
`cut_x=0.3`/`cut_y=0.3` を保持している。K.2 の生成条件（無条件生成）をそのまま適用すると、
この既存fixtureに対しても小棚AABBが計算され、T-006の既存アサーション
（`shelf_boxes == []`、`effective_volume` 誤差 <1%）と矛盾する。

→ `container_space.py` 側にfixture救済のための特別分岐は追加しない。矛盾は **fixture側の不整合**
として扱い、T-007のテスト準備セッションで `_box_cdict` の既定値を `cut_x=0.0, cut_y=0.0` に修正し、
真の「cutなし直方体」を表すfixtureへ更新する（`shelf=False` は維持）。この修正は
`tests/test_container_space.py` の編集権限を持つテスト準備セッションで行う。

### K.6 cut plane構築の一般化（T-006ロジックの拡張、変更なし）

T-006で実装済みの `_axis_alignment` による半空間分類（軸整列→`inner_min_rel`/`inner_max_rel`、
非軸整列→`cut_planes`）は `cdict["points"]`/`["n_vecs"]` に対して汎用であり、cutの有無に
よらずそのまま適用する。追加の特別処理は不要。

### K.7 floor_z / ceil_z の算出規則

格子セル中心 `(x_c, y_c)`（コンテナ相対）について、各 `cut_plane (normal_rel=(nx,ny,nz), d)` を
次のように分類し反映する：

* `nz < -EPS_GEOM`（下向き成分を持つ面）: `normal·(x_c,y_c,z) <= d` を `z` について解くと
  `z >= (d - nx*x_c - ny*y_c) / nz`（負の `nz` で除するため不等号反転）。この下限を
  `floor_z[i,j]` の候補とし、`inner_min_rel[2]` との `max` を取る。
* `nz > +EPS_GEOM`（上向き成分を持つ面）: 同様に `z <= (d - nx*x_c - ny*y_c) / nz`。
  この上限を `ceil_z[i,j]` の候補とし、`inner_max_rel[2]` との `min` を取る。
* `ceil_z[i,j]` はさらに、そのセルのXY範囲と交差する `shelf_boxes` の下面 `z`（`box_min[2]`）
  とも `min` を取る。

**セル中心座標・格子添字の規約**（T-006実装 `container_space.py:114-119` を正式仕様として明文化。
矛盾なし、変更不要）：

```python
nx = max(1, round(size[0] / cell))   # size = inner_max_rel - inner_min_rel（T-006既存式）
ny = max(1, round(size[1] / cell))

x_c[i] = inner_min_rel[0] + (i + 0.5) * cell   # i = 0, ..., nx-1
y_c[j] = inner_min_rel[1] + (j + 0.5) * cell   # j = 0, ..., ny-1
```

* 有効index範囲は `i ∈ {0,...,nx-1}`, `j ∈ {0,...,ny-1}`（`floor_z`/`ceil_z`/`height` の shape
  `(nx,ny)` と一致、T-006のまま）。
* **端数セルの扱い**: `round()` により `nx*cell` は `size[0]` と厳密には一致しない場合がある
  （最大 `cell/2` の差）。`effective_volume()` は全セルを `cell×cell` として扱い、この端数分の
  面積補正は行わない（T-006 `effective_volume()` の既存実装のまま、変更不要）。
* **セル中心が `inner_max_rel` を超えないことの証明**: `nx = round(size/cell)` の定義より
  `|nx - size/cell| <= 0.5` ⇒ `nx <= size/cell + 0.5` ⇒ `(nx-0.5)*cell <= size` ⇒
  `x_c[nx-1] = inner_min_rel[0] + (nx-0.5)*cell <= inner_min_rel[0] + size[0] = inner_max_rel[0]`
  （Y軸も同様）。等号は `size/cell` がちょうど整数+0.5のときのみ成立。したがって全セル中心は
  常に `[inner_min_rel, inner_max_rel]` の範囲内に収まり、境界超過に対する特別なクランプ処理は
  不要。

### K.8 `normal_z ≈ 0`（`|nz| <= EPS_GEOM`）の面の扱い

z方向を拘束しない面（XY方向のみの制約）。`floor_z`/`ceil_z` の算出式には**寄与させない**
（K.7の分類から除外する）。格子は絞り込み専用であり（本書規約・実装詳細仕様書§2）、
このような面によるXY方向の除外は `contains_oriented_box` の8頂点×全`cut_planes`半空間判定
でのみ最終的に行う。格子段階でこれを無視しても、格子は事前絞り込み用のため安全側
（誤って除外しない）に倒れるだけであり、健全性は損なわれない。

### K.9 golden fixtureの生成方針（T-007テスト準備セッションで実施）

* 幾何（`points`/`n_vecs`）: `ground_handling/utils.py::write_open_cut_corner_cup_obj` は
  PyBullet非依存の純関数。固定の `cut_x/cut_y/thickness/length/height/width` を渡して
  **実際に呼び出し**、返り値をgolden入力とする（読解結果の転記ではなく、公式関数の直接呼び出し
  による値取得。コピーではない）。
* 期待 `floor_z`/`ceil_z`: `container_space.py` を経由せず、K.7の式を独立実装
  （別スクリプトまたはテスト内のローカル計算）してgolden配列を作る。
* shelf AABB: K.2/K.3/K.4の式に同じ固定値を代入した独立計算値をgoldenとする。
* `effective_volume`（棚ありケース）の許容誤差: **本文書では定義しない**。テスト準備セッションで
  `cdict["volume"]`（または`evaluator.py`の式）との実測差を確認し、格子離散化誤差から説明可能な
  範囲かを検証したうえで許容値を決定する。単一区間の `floor_z`/`ceil_z` では表現できない空間
  （例：棚の上下に有効空間が分離するセル）が確認された場合は、その時点で付録D形式により停止し
  人間に報告する。

### K.10 `effective_volume()` の責務再定義（実測調査結果）

固定パラメータ（`length=height=width=1.00`, `thickness=0.02`, `cut_x=cut_y=0.20`, `buffer=0.02`,
`shelf=False`）で `write_open_cut_corner_cup_obj`/`aff` を直接呼び出し、K.6/K.7 の式を
`container_space.py` を使わず独立実装して検証したところ、格子（`floor_z`/`ceil_z` 単一区間）から
積分した体積は `cdict["volume"]`（公式体積式）と構造的に一致しないことが判明した。

**実測値**（このfixtureにおける相対誤差、公式体積式 `base − cut − small_shelf`＝0.865344 m³ 基準）：

| 反映内容 | 格子積分体積 | 相対誤差 |
| --- | --- | --- |
| cut床上昇のみ | 0.885831 m³ | 約2.37% |
| cut床上昇＋小棚ceilキャップ | 0.793671 m³ | 約8.28% |

**原因は2つ、独立に存在する**：

1. **入口面（door側）の非対称性**（棚の有無によらず発生）：公式体積式は `inner_width = width - 2*thickness`
   （両側とも壁厚を引く前提）を使うが、実ジオメトリ（`points`/`n_vecs`）では入口面（-Y側）は開口のため
   壁厚が引かれず、Y方向の実スパンは `width - thickness` になる（実測: 0.98 vs 公式式の0.96）。
   この差だけで約2.37%の乖離が生じる。
2. **棚のceilキャップによる上方空間喪失**（K.2の小棚無条件生成の帰結）：`ceil_z` を棚下面で
   キャップする設計（§4.2・K.7）では、棚は薄い障害物（厚さ`thickness`のみ）であるにもかかわらず、
   単一区間の `floor_z`/`ceil_z` モデルでは棚の**上側空間全体**が表現から失われる。実測で
   0.09216 m³（公式体積の約10.65%）の喪失。この喪失は格子解像度を上げても解消しない
   （量子化誤差ではなく、単一区間モデルの表現力の限界）。

**採用した設計判断**（人間承認・2026-07-17）：

* `effective_volume()` は「`floor_z`/`ceil_z` 単一区間格子上の近似・診断用体積」と責務を明確化する。
  公式fillスコアの分母には使わない（fill分母は `evaluator.py`／転記済み公式値を正とする既存方針を
  維持・強化）。
* `cdict["volume"]` との <1% 一致は、cut・棚を含むfixtureのDoDから外す。代わりに、golden の
  `floor_z`/`ceil_z` から独立計算した格子積分値（`sum(max(ceil_z-floor_z,0)*cell*cell)`）との
  一致を検証する（＝実装が仕様どおりの格子積分を行っているかの確認に限定する）。
* `cdict["volume"]` との差は診断値として記録するのみで合否条件にしない。
* 直方体基準ケース（`cut_x=0`・`cut_y=0`・shelf=False・小棚クリップ後ゼロ体積）に限っては、
  上記いずれの原因も発生しないため、引き続き解析的内壁体積との相対誤差 <1% を要求する。
* 棚上方空間を表現できない制約は、複数区間（バンド）格子への拡張なしに解消できないため、
  T-007/T-008のスコープには含めない既知の制約として記録する（将来の改善候補）。
* 棚体積のみを別経路で単純減算する代替案（`ceil_z`キャップと独立に体積だけ補正する方式）は、
  同一関数内に異なる意味の空間モデルを二重化することになるため今回は不採用。

---

## 更新履歴

| 日付 | 内容 |
| --- | --- |
| 2026-07-17 | 初版（T-002）。§A〜H 転記表、§I 不一致一覧、§J 質問リスト |
| 2026-07-17 | T-006 Session A：§I-7 追記（`buffer` キー不在の解決方針）。`実装詳細仕様書.md` §4.2 を同方針で修正 |
| 2026-07-17 | T-007 仕様補正：§K 追加（cut plane の floor_z/ceil_z 反映規則、小棚・大棚AABBの復元式とクリップ・正体積フィルタ、生成条件は `cut_planes` に依存させない設計判断、T-006 fixture の `cut_x`/`cut_y` 不整合はfixture側の問題としてテスト準備セッションで修正する方針、golden fixture 生成方針）。`実装詳細仕様書.md` §4.2・§6 を同方針で修正 |
| 2026-07-17 | T-007 仕様再補正：§K.10 追加。実測調査により棚ありfixtureの`effective_volume`が`cdict["volume"]`と<1%一致しない構造的原因（入口面非対称性 約2.37%、棚ceilキャップによる上方空間喪失 約10.65%）を特定。`effective_volume()`の責務を近似・診断用に再定義し、DoDを直方体基準ケース（<1%維持）とcut・棚ケース（golden格子積分との一致、`cdict["volume"]`一致は不要）に分離。`実装詳細仕様書.md` §4.2・§6 を同方針で修正 |
| 2026-07-17 | T-007 仕様追記：§K.7 にセル中心座標規約（`x_c[i]=inner_min_rel[0]+(i+0.5)*cell`等）・`nx`/`ny`算出式（T-006既存式を正式仕様化）・端数セルの面積補正なし方針・セル中心が`inner_max_rel`を超えないことの数学的証明を追記。T-006実装との矛盾なし。`実装詳細仕様書.md` §4.2 を同方針で修正 |
| 2026-07-17 | T-012 計画中に発覚した不一致を訂正：§E に `spacing` が agent I/F に不在である旨を追記。§I-8 を新設し、コンテナ原点世界X（`offset_x`）の取得元を `i * spacing`（`init_states` に不在の値）ではなく `cdict["center"][0]`（`containers.py:61,66,238–240` により `center[0]==offset_x` が恒等的に成立）とする解決方針を記録。`実装詳細仕様書.md` §3.1/§4.2/§4.4・`初期検討_実装手順書.md` §1.2 を同方針で修正 |
