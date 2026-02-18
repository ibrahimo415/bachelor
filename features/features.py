import cv2
import numpy as np
from PIL import Image
from skimage import img_as_float, restoration

def resize_image_smart(img_pil, max_side=1024):
    w, h = img_pil.size
    if max(h, w) <= max_side:
        return img_pil, False
    scale = max_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img_resized, True

def compute_features(image_path, max_side=1024):
    try:
        img_rgb_pil = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    img_rgb_pil, was_resized = resize_image_smart(img_rgb_pil, max_side=max_side)
    img_rgb = np.asarray(img_rgb_pil, dtype=np.uint8)

    gray_u8 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    img_float = img_as_float(img_rgb)
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = np.hypot(np.std(rg), np.std(yb))
    mean_root = np.hypot(np.mean(rg), np.mean(yb))
    colorfulness = float(std_root + 0.3 * mean_root)

    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    sharpness = float(lap.var())

    noise = float(restoration.estimate_sigma(img_float, channel_axis=-1, average_sigmas=True))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "sharpness": sharpness,
        "noise": noise,
        "was_resized": bool(was_resized)
    }
