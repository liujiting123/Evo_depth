import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


IMAGE_NAMES = ("image_1_rgb.png", "image_2_rgb.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Create mild visual perturbations for an EvoDepth episode dir.")
    parser.add_argument("--base-dir", required=True, help="Episode dir containing obs_*/image_*.png and state.json.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    x = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.clip((x**gamma) * 255.0, 0.0, 255.0).astype(np.uint8)


def apply_shadow(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 0.92 + 0.10 * ((xx / max(w - 1, 1)) * 0.65 + (yy / max(h - 1, 1)) * 0.35)
    out = rgb.astype(np.float32) * grad[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def transform_image(image: Image.Image, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "original":
        return image
    if variant == "bright_p08":
        return ImageEnhance.Brightness(image).enhance(1.08)
    if variant == "bright_m08":
        return ImageEnhance.Brightness(image).enhance(0.92)
    if variant == "contrast_p10":
        return ImageEnhance.Contrast(image).enhance(1.10)
    if variant == "contrast_m10":
        return ImageEnhance.Contrast(image).enhance(0.90)
    if variant == "gamma_0p90":
        return Image.fromarray(apply_gamma(np.asarray(image), 0.90), mode="RGB")
    if variant == "gamma_1p10":
        return Image.fromarray(apply_gamma(np.asarray(image), 1.10), mode="RGB")
    if variant == "shadow_diag":
        return Image.fromarray(apply_shadow(np.asarray(image)), mode="RGB")
    raise ValueError(f"unknown variant: {variant}")


def image_diff_stats(base: Image.Image, aug: Image.Image) -> dict[str, float]:
    a = np.asarray(base.convert("RGB"), dtype=np.int16)
    b = np.asarray(aug.convert("RGB"), dtype=np.int16)
    diff = np.abs(a - b)
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "pct_any_channel_gt_10": float((diff.max(axis=2) > 10).mean() * 100.0),
        "pct_any_channel_gt_20": float((diff.max(axis=2) > 20).mean() * 100.0),
    }


def copy_sidecars(src_obs: Path, dst_obs: Path):
    for name in ("state.json", "meta.json"):
        src = src_obs / name
        if src.exists():
            shutil.copy2(src, dst_obs / name)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_root = Path(args.output_root)
    variants = [
        "original",
        "bright_p08",
        "bright_m08",
        "contrast_p10",
        "contrast_m10",
        "gamma_0p90",
        "gamma_1p10",
        "shadow_diag",
    ]

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    obs_dirs = sorted(path for path in base_dir.glob("obs_*") if path.is_dir())
    if not obs_dirs:
        raise FileNotFoundError(f"no obs_* dirs under {base_dir}")

    summary = {"base_dir": str(base_dir), "output_root": str(output_root), "variants": {}}
    for variant in variants:
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_stats = []
        for obs_dir in obs_dirs:
            out_obs = variant_dir / obs_dir.name
            out_obs.mkdir(parents=True, exist_ok=True)
            copy_sidecars(obs_dir, out_obs)
            for image_name in IMAGE_NAMES:
                src_image = Image.open(obs_dir / image_name).convert("RGB")
                aug_image = transform_image(src_image, variant)
                aug_image.save(out_obs / image_name)
                variant_stats.append({"obs": obs_dir.name, "image": image_name, **image_diff_stats(src_image, aug_image)})
        summary["variants"][variant] = {
            "dir": str(variant_dir),
            "mean_abs_mean": float(np.mean([item["mean_abs"] for item in variant_stats])),
            "max_abs_max": float(np.max([item["max_abs"] for item in variant_stats])),
            "pct_gt_10_mean": float(np.mean([item["pct_any_channel_gt_10"] for item in variant_stats])),
            "pct_gt_20_mean": float(np.mean([item["pct_any_channel_gt_20"] for item in variant_stats])),
        }

    with (output_root / "augmentation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
