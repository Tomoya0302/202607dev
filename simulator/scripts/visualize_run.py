"""1タスク・1 agent設定を実行し、PyBulletのstep経過をGIFとして保存する（可視化専用、
提出物とは無関係の診断ツール）。

`scripts/bench_run.py`の実行経路（`GroundHandlingEnv`/`TimedAgentRunner`、
get_init_states→optimize→policy loop）をそのまま再現しつつ、`env.py`に既に存在する
（`config["visualizer"]["vis"]`で有効化する）ステップごとのPNGスナップショット機構
（`env.reset`/`env.step`内、`Camera.get_rgb_image`）を使い、実行後にPNG連番をGIFへ
まとめる。カメラパラメータはタスクJSONの`camera`（観測用、target_pos/distance/fov等）を
そのまま流用し、`visualizer.camera`のyaw/pitchのみ上書きする（元タスクの意図した
可視化アングルを尊重する）。

使用例（design文書§11.9、Sequence Tripleの実配置を見る診断用）:
    python -m scripts.visualize_run --config-path artifacts/bench_tasks/family1/f1_0000.json \\
        --extra-env GH_SEQ_TRIPLE=1 --extra-env GH_SEQ_TRIPLE_FORCE=1 \\
        --out artifacts/visualizations/seq_triple_f1_0000_packing.gif
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())


def run_one(config_path: str, module_path: str, agent_module: str, agent_class: str,
            out_path: str, width: int, height: int, fps: int, distance_mul: float) -> dict:
    from PIL import Image

    from src.ground_handling.agent_factory import AgentFactory
    from src.ground_handling.env import GroundHandlingEnv
    from src.ground_handling.runner import TimedAgentRunner

    with open(config_path) as f:
        config = json.load(f)
    task_id, task_config = next(iter(config.items()))

    # 観測用カメラ(target_pos/distance/fov/near_val/far_val)をそのまま流用し、
    # 解像度とyaw/pitchだけ可視化用に上書きする（タスクJSONの`visualizer.camera`が
    # 意図しているアングルを尊重する、§11.9）。
    base_cam = dict(task_config["camera"])
    vis_yaw = task_config.get("visualizer", {}).get("camera", {}).get("yaw", 30)
    vis_pitch = task_config.get("visualizer", {}).get("camera", {}).get("pitch", -20)
    base_cam.update({
        "yaw": vis_yaw, "pitch": vis_pitch,
        "img_width": width, "img_height": height,
        "distance": float(base_cam.get("distance", 3.0)) * distance_mul,
    })
    task_config = dict(task_config)
    task_config["visualizer"] = {"vis": True, "camera": base_cam}

    agent_factory = AgentFactory(module_name=agent_module, class_name=agent_class, module_path=module_path)
    env = None
    runner = None
    image_dir = None
    try:
        env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
        image_dir = env.image_dir
        env.reset_settings()
        init_states = env.get_init_states()

        allowed_methods = task_config["agent"]["allowed_methods"]
        max_mem = task_config["agent"].get("max_mem", 4)
        runner = TimedAgentRunner(agent_factory=agent_factory, allowed_methods=allowed_methods,
                                  max_mem=max_mem, verbose=False)
        runner.call("get_init_states", time_out_sec=task_config["agent"]["init_timeout"],
                    fallback=None, init_states=init_states)

        optimize_time = 0.0
        if env.optimize:
            item_list = env.get_info_for_optimization()
            optimized_order, opt_elapsed = runner.call(
                "optimize", time_out_sec=task_config["agent"]["optimization_timeout"],
                fallback=list(env.stream_manager.all_indices), item_list=item_list)
            optimize_time = opt_elapsed
            if not env.set_item_order(optimized_order):
                return {"config": os.path.basename(config_path), "status": "format_error",
                        "message": "invalid optimize order"}
        env.reset_item_stream()

        obs, info = env.reset(seed=42)
        terminated = truncated = False
        n_steps = 0
        while not terminated and not truncated:
            action, _p_elapsed = runner.call("policy", time_out_sec=task_config["agent"]["policy_timeout"],
                                              fallback=env.action_space.sample(), observation=obs)
            obs, reward, terminated, truncated, info = env.step(action)
            n_steps += 1

        fill_score, out_items = env.evaluator.calculate_fill_rate(env.container_manager.containers)
        num_total = env.num_total_items
        num_placed_items = sum(len(c.packed_items) for c in env.container_manager.containers) / max(1, num_total)

        print(f"DEBUG cwd={os.getcwd()} image_dir={image_dir} exists={os.path.exists(image_dir)}",
              file=sys.stderr)
        if os.path.isdir(image_dir):
            print(f"DEBUG listing={sorted(os.listdir(image_dir))}", file=sys.stderr)
        frames = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
        images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in frames]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if images:
            duration_ms = int(1000 / max(1, fps))
            images[0].save(out_path, save_all=True, append_images=images[1:],
                            duration=duration_ms, loop=0)

        return {"config": os.path.basename(config_path), "status": "success",
                "fill_score": float(fill_score), "num_placed_items": float(num_placed_items),
                "optimization_time": float(optimize_time), "n_steps": n_steps,
                "n_frames": len(images), "out_path": out_path}
    except Exception as exc:
        return {"config": os.path.basename(config_path), "status": "exec_exception", "message": str(exc)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        if image_dir is not None and os.path.exists(out_path):
            shutil.rmtree(os.path.dirname(image_dir), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--agent-class", default="Agent")
    parser.add_argument("--out", required=True, help="出力GIFパス")
    parser.add_argument("--extra-env", action="append", default=[],
                        help="KEY=VALUE形式（複数指定可）。agent実行前に環境変数として設定する")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--distance-mul", type=float, default=1.0,
                        help="観測用カメラのdistanceに掛ける倍率（画角を広げたい場合に上げる）")
    args = parser.parse_args()

    for kv in args.extra_env:
        k, _, v = kv.partition("=")
        os.environ[k] = v

    agent_module_path = ".".join(args.module_path.split("/")) + "agent"
    t0 = time.time()
    result = run_one(args.config_path, args.module_path, agent_module_path, args.agent_class,
                     args.out, args.width, args.height, args.fps, args.distance_mul)
    result["wall_time"] = time.time() - t0
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
