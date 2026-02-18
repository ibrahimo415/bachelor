import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import img_as_float, restoration
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision.transforms import ToTensor


def resize_cv2_inter_area(img_bgr: np.ndarray, max_side: int = 1024):
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_side:
        return img_bgr, False
    scale = max_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized, True


def resize_pil_lanczos(img: Image.Image, max_side: int = 1024):
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS), True
    return img, False


def thumbnail_pil_lanczos(img: Image.Image, max_side: int = 1024):
    out = img.copy()
    old_size = out.size
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out, out.size != old_size


def handcrafted_from_rgb_u8(rgb_u8: np.ndarray):
    gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    img_float = img_as_float(rgb_u8)
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.hypot(np.std(rg), np.std(yb))
    mean_root = np.hypot(np.mean(rg), np.mean(yb))
    colorfulness = float(std_root + 0.3 * mean_root)

    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = float(lap.var())

    noise = float(restoration.estimate_sigma(img_float, channel_axis=-1, average_sigmas=True))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "sharpness": sharpness,
        "noise": noise,
    }


def build_clip_metric():
    all_prompts = (
        "quality", "brightness", "noisiness", "colorfullness",
        "sharpness", "contrast", "complexity", "natural",
        "happy", "scary", "new", "warm", "real",
        "beautiful", "lonely", "relaxing"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    metric = CLIPImageQualityAssessment(prompts=all_prompts, data_range=255.0).to(device).eval()
    return metric, device, all_prompts


def clip_scores_from_pil(metric, device, prompts, img_pil: Image.Image):
    img_tensor = (ToTensor()(img_pil) * 255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = metric(img_tensor)
    return {f"clip_{p}": float(scores[p].detach().cpu().view(-1)[0].item()) for p in prompts}


def print_feature_table(features_a, features_b, label_a, label_b):
    keys = ["brightness", "contrast", "colorfulness", "sharpness", "noise"]
    print("\nFeature comparison (same image):")
    print(f"{'feature':<14} {label_a:>14} {label_b:>14} {'delta':>14} {'delta_%':>10}")
    print("-" * 70)
    for key in keys:
        a = float(features_a[key])
        b = float(features_b[key])
        delta = b - a
        delta_pct = (delta / a * 100.0) if abs(a) > 1e-12 else float("nan")
        print(f"{key:<14} {a:>14.6f} {b:>14.6f} {delta:>14.6f} {delta_pct:>10.3f}")


def print_clip_table(scores_a, scores_b, label_a, label_b):
    keys = sorted(scores_a.keys())
    print("\nCLIP-IQA comparison (same image):")
    print(f"{'prompt':<20} {label_a:>14} {label_b:>14} {'delta':>14} {'delta_%':>10}")
    print("-" * 80)
    for key in keys:
        a = float(scores_a[key])
        b = float(scores_b[key])
        delta = b - a
        delta_pct = (delta / a * 100.0) if abs(a) > 1e-12 else float("nan")
        print(f"{key:<20} {a:>14.6f} {b:>14.6f} {delta:>14.6f} {delta_pct:>10.3f}")


def print_pixel_diff(img_a_u8, img_b_u8, label_a, label_b):
    if img_a_u8.shape != img_b_u8.shape:
        print("\nPixel diff skipped: resized shapes differ.")
        print(f"{label_a} RGB shape: {img_a_u8.shape}")
        print(f"{label_b} RGB shape: {img_b_u8.shape}")
        return

    diff = np.abs(img_a_u8.astype(np.int16) - img_b_u8.astype(np.int16))
    mae = float(diff.mean())
    max_abs = int(diff.max())
    changed_pixels_pct = float((diff > 0).any(axis=2).mean() * 100.0)

    print("\nPixel-level difference after resize:")
    print(f"MAE over RGB channels: {mae:.6f}")
    print(f"Max abs channel diff:  {max_abs}")
    print(f"Pixels with any change: {changed_pixels_pct:.3f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare handcrafted features across OpenCV resize, PIL resize, and PIL thumbnail."
    )
    parser.add_argument("image_path", help="Path to one input image")
    parser.add_argument("--max-side", type=int, default=1024, help="Maximum image side (default: 1024)")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP-IQA comparison")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image_label = os.path.basename(args.image_path)

    img_bgr_orig = cv2.imread(args.image_path)
    if img_bgr_orig is None:
        raise RuntimeError(f"Could not read image via OpenCV: {args.image_path}")
    img_h, img_w = img_bgr_orig.shape[:2]

    img_cv2_bgr, cv2_resized = resize_cv2_inter_area(img_bgr_orig, max_side=args.max_side)
    img_cv2_rgb = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
    features_cv2 = handcrafted_from_rgb_u8(img_cv2_rgb)

    img_pil_orig = Image.open(args.image_path).convert("RGB")
    img_pil_resize, pil_resize_resized = resize_pil_lanczos(img_pil_orig, max_side=args.max_side)
    img_pil_thumb, pil_thumb_resized = thumbnail_pil_lanczos(img_pil_orig, max_side=args.max_side)
    img_pil_resize_rgb = np.asarray(img_pil_resize, dtype=np.uint8)
    img_pil_thumb_rgb = np.asarray(img_pil_thumb, dtype=np.uint8)
    features_pil_resize = handcrafted_from_rgb_u8(img_pil_resize_rgb)
    features_pil_thumb = handcrafted_from_rgb_u8(img_pil_thumb_rgb)

    print(f"Image: {args.image_path}")
    print(f"Original size: {img_w}x{img_h} (W x H)")
    print(f"OpenCV resized: {img_cv2_rgb.shape[1]}x{img_cv2_rgb.shape[0]} | was_resized={cv2_resized}")
    print(
        f"PIL resize:     {img_pil_resize_rgb.shape[1]}x{img_pil_resize_rgb.shape[0]} "
        f"| was_resized={pil_resize_resized}"
    )
    print(
        f"PIL thumbnail:  {img_pil_thumb_rgb.shape[1]}x{img_pil_thumb_rgb.shape[0]} "
        f"| was_resized={pil_thumb_resized}"
    )

    print(f"\n=== OpenCV vs PIL resize | image={image_label} ===")
    print_pixel_diff(img_cv2_rgb, img_pil_resize_rgb, "OpenCV", "PIL resize")
    print_feature_table(features_cv2, features_pil_resize, "opencv", "pil_resize")

    print(f"\n=== PIL resize vs PIL thumbnail | image={image_label} ===")
    print_pixel_diff(img_pil_resize_rgb, img_pil_thumb_rgb, "PIL resize", "PIL thumbnail")
    print_feature_table(features_pil_resize, features_pil_thumb, "pil_resize", "pil_thumb")

    if not args.skip_clip:
        metric, device, prompts = build_clip_metric()
        clip_resize = clip_scores_from_pil(metric, device, prompts, img_pil_resize)
        clip_thumb = clip_scores_from_pil(metric, device, prompts, img_pil_thumb)

        print(f"\n=== CLIP-IQA: PIL resize vs PIL thumbnail | image={image_label} ===")
        print_clip_table(clip_resize, clip_thumb, "pil_resize", "pil_thumb")


if __name__ == "__main__":
    main()
