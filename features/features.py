import cv2
import numpy as np
from skimage import img_as_float, restoration

def resize_image_smart(img_bgr, max_side=1024):
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_side:
        return img_bgr, False
    scale = max_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized, True

def compute_features(image_path, max_side=1024):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_bgr, was_resized = resize_image_smart(img_bgr, max_side=max_side)

    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    colorfulness = float(np.std(rg) + np.std(yb) + 0.3 * (np.mean(rg) + np.mean(yb)))

    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
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
