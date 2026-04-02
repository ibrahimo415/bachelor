import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from skimage.restoration import estimate_sigma
from torchmetrics.multimodal import CLIPImageQualityAssessment


IMG_PATH = "953417.jpg"
MAX_SIDE = 1024
CLIP_PROMPTS = ("quality", "brightness", "sharpness", "noisiness")


device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


def resize_max_side_lanczos(img: Image.Image, max_side: int):
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img, True
    return img, False


def handcrafted_from_pil(img_pil: Image.Image):
    rgb_u8 = np.asarray(img_pil, dtype=np.uint8)              # (H,W,3) RGB
    gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)        # (H,W)
    gray = gray_u8.astype(np.float32) / 255.0

    brightness = float(gray.mean())
    contrast   = float(gray.std())

    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = float(lap.var())

    rgb_f = rgb_u8.astype(np.float32) / 255.0
    noise = float(estimate_sigma(rgb_f, channel_axis=-1, average_sigmas=True))

    return {
        "hand_brightness": brightness,
        "hand_contrast": contrast,
        "hand_sharpness": sharpness,
        "hand_noise": noise,
    }


def main():
    img0 = Image.open(IMG_PATH).convert("RGB")
    print("orig size:", img0.size)

    imgL, was_resized = resize_max_side_lanczos(img0, MAX_SIDE)
    print("lanczos size:", imgL.size, "| was_resized:", was_resized)

    rgb_u8 = np.asarray(imgL, dtype=np.uint8)
    checksum = int(rgb_u8.sum())
    print("pixel checksum (sum of uint8 RGB):", checksum)

    hand = handcrafted_from_pil(imgL)

    metric = CLIPImageQualityAssessment(
        model_name_or_path="clip_iqa",
        prompts=CLIP_PROMPTS,
        data_range=1.0
    ).to(device).eval()

    x01 = ToTensor()(imgL).unsqueeze(0).to(device)
    print("clip tensor shape:", tuple(x01.shape))

    with torch.inference_mode():
        clip = metric(x01)

    clip_scores = {f"clip_{k}": float(clip[k].cpu().item()) for k in CLIP_PROMPTS}

    print("\n=== HANDCRAFTED (PIL-LANCZOS resized) ===")
    for k, v in hand.items():
        print(f"{k:15s}", v)

    print("\n=== CLIP-IQA (PIL-LANCZOS resized) ===")
    for k, v in clip_scores.items():
        print(f"{k:15s}", v)

    print("\nconfirm same pixels used:")
    print("handcrafted input shape:", rgb_u8.shape, "(H,W,3)")

if __name__ == "__main__":
    main()
