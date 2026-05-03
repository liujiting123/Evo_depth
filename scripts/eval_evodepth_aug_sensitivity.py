import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import websockets
from PIL import Image


IMAGE_NAMES = ("image_1_rgb.png", "image_2_rgb.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EvoDepth action sensitivity to mild image perturbations.")
    parser.add_argument("--server-uri", default="ws://localhost:9000")
    parser.add_argument("--input-root", required=True, help="Root containing variant/obs_*/image_*.png dirs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-variant", default="original")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--obs-indices", default="0,2,4,6", help="Comma-separated obs indices, or 'all'.")
    parser.add_argument("--base-repeats", type=int, default=3)
    parser.add_argument("--variant-repeats", type=int, default=1)
    parser.add_argument("--prompt", default="pick green block on yellow block")
    parser.add_argument("--train-resize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def discover_obs_dirs(variant_dir: Path, obs_indices: str) -> list[Path]:
    obs_dirs = sorted(path for path in variant_dir.glob("obs_*") if path.is_dir())
    if obs_indices == "all":
        return obs_dirs
    keep = {int(item) for item in obs_indices.split(",") if item.strip()}
    return [path for idx, path in enumerate(obs_dirs) if idx in keep]


def load_image_for_server(path: Path, train_resize: bool) -> np.ndarray:
    if train_resize:
        image = Image.open(path).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        rgb = np.asarray(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return bgr


def load_state(obs_dir: Path) -> np.ndarray:
    with (obs_dir / "state.json").open("r") as f:
        state_data = json.load(f)
    if isinstance(state_data, dict):
        state_data = state_data["state"]
    return np.asarray(state_data[:6], dtype=np.float32)


def build_request(obs_dir: Path, args) -> tuple[dict, np.ndarray]:
    image_1 = load_image_for_server(obs_dir / IMAGE_NAMES[0], args.train_resize)
    image_2 = load_image_for_server(obs_dir / IMAGE_NAMES[1], args.train_resize)
    dummy = np.zeros_like(image_1, dtype=np.uint8)
    state = load_state(obs_dir)
    pad_dim = 24 - state.shape[0]
    request = {
        "image": [image_1.tolist(), image_2.tolist(), dummy.tolist()],
        "image_mask": [1, 1, 0],
        "state": np.pad(state, (0, pad_dim), constant_values=0).astype(float).tolist(),
        "action_mask": [[1] * state.shape[0] + [0] * pad_dim],
        "prompt": args.prompt,
    }
    return request, state


def chunk_metrics(actions: np.ndarray, states: np.ndarray) -> dict[str, float]:
    actions_6d = actions[:, :, :6]
    states_6d = states[:, None, :6]
    span = actions_6d.max(axis=1) - actions_6d.min(axis=1)
    delta_last = actions_6d[:, -1, :] - states_6d[:, 0, :]
    delta_10 = actions_6d[:, min(10, actions_6d.shape[1] - 1), :] - states_6d[:, 0, :]
    return {
        "span_l2_mean": float(np.linalg.norm(span, axis=1).mean()),
        "span_l2_max": float(np.linalg.norm(span, axis=1).max()),
        "delta10_l2_mean": float(np.linalg.norm(delta_10, axis=1).mean()),
        "delta_last_l2_mean": float(np.linalg.norm(delta_last, axis=1).mean()),
    }


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a[:, :, :6] - b[:, :, :6])
    return {
        "max_abs_6d": float(diff.max()),
        "mean_abs_6d": float(diff.mean()),
        "rms_abs_6d": float(np.sqrt((diff**2).mean())),
        "per_obs_max_abs_6d": [float(x) for x in diff.max(axis=(1, 2))],
    }


async def infer_variant(ws, variant: str, variant_dir: Path, obs_dirs: list[Path], repeat_i: int, args, output_dir: Path):
    actions = []
    states = []
    obs_names = []
    for obs_dir in obs_dirs:
        request, state = build_request(obs_dir, args)
        await ws.send(json.dumps(request))
        response = await ws.recv()
        action = np.asarray(json.loads(response), dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 24:
            raise ValueError(f"bad action shape for {variant}/{obs_dir.name}: {action.shape}")
        actions.append(action)
        states.append(state)
        obs_names.append(obs_dir.name)
        print(f"[{variant} r{repeat_i}] {obs_dir.name} action_shape={action.shape}", flush=True)

    actions_arr = np.stack(actions, axis=0)
    states_arr = np.stack(states, axis=0)
    run_name = f"{variant}_r{repeat_i}"
    np.save(output_dir / f"{run_name}_actions.npy", actions_arr)
    np.save(output_dir / f"{run_name}_states.npy", states_arr)
    return run_name, {
        "variant": variant,
        "repeat": repeat_i,
        "variant_dir": str(variant_dir),
        "obs_names": obs_names,
        "actions_npy": str(output_dir / f"{run_name}_actions.npy"),
        "states_npy": str(output_dir / f"{run_name}_states.npy"),
        "metrics": chunk_metrics(actions_arr, states_arr),
    }, actions_arr


async def main_async(args):
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variants is None:
        variants = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    else:
        variants = args.variants
    variants = [args.base_variant] + [variant for variant in variants if variant != args.base_variant]

    base_obs_dirs = discover_obs_dirs(input_root / args.base_variant, args.obs_indices)
    if not base_obs_dirs:
        raise FileNotFoundError(f"no selected obs dirs under {input_root / args.base_variant}")

    runs = {}
    actions_by_run = {}
    async with websockets.connect(args.server_uri, max_size=100_000_000) as ws:
        for repeat_i in range(args.base_repeats):
            run_name, run, actions = await infer_variant(
                ws, args.base_variant, input_root / args.base_variant, base_obs_dirs, repeat_i, args, output_dir
            )
            runs[run_name] = run
            actions_by_run[run_name] = actions

        for variant in variants:
            if variant == args.base_variant:
                continue
            obs_dirs = discover_obs_dirs(input_root / variant, args.obs_indices)
            if [path.name for path in obs_dirs] != [path.name for path in base_obs_dirs]:
                raise ValueError(f"obs dirs mismatch for {variant}")
            for repeat_i in range(args.variant_repeats):
                run_name, run, actions = await infer_variant(
                    ws, variant, input_root / variant, obs_dirs, repeat_i, args, output_dir
                )
                runs[run_name] = run
                actions_by_run[run_name] = actions

    base_names = [f"{args.base_variant}_r{i}" for i in range(args.base_repeats)]
    baseline_pairwise = {}
    for i, name_i in enumerate(base_names):
        for name_j in base_names[i + 1 :]:
            baseline_pairwise[f"{name_i}__{name_j}"] = diff_metrics(actions_by_run[name_i], actions_by_run[name_j])

    variant_vs_base = {}
    base_ref = actions_by_run[base_names[0]]
    for run_name, actions in actions_by_run.items():
        if run_name == base_names[0]:
            continue
        variant_vs_base[run_name] = diff_metrics(base_ref, actions)

    summary = {
        "server_uri": args.server_uri,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "obs_indices": args.obs_indices,
        "train_resize": args.train_resize,
        "runs": runs,
        "baseline_pairwise": baseline_pairwise,
        "variant_vs_base_r0": variant_vs_base,
    }
    with (output_dir / "sensitivity_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
