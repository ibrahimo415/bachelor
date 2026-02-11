import os

# =========================================================
# 0) CACHE-UMLEITUNG (MUSS GANZ OBEN STEHEN!)
#    -> muss VOR torch/pyiqa/transformers imports ausgeführt werden
# =========================================================
BASE = "/data/stud/2026-BA-ibrahim_osman"

os.environ["HF_HOME"] = os.path.join(BASE, "huggingface_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE, "huggingface_cache", "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE, "huggingface_cache", "transformers")
os.environ["XDG_CACHE_HOME"] = os.path.join(BASE, "huggingface_cache", "xdg")

os.environ["TORCH_HOME"] = os.path.join(BASE, "torch_cache")
os.environ["TMPDIR"] = os.path.join(BASE, "torch_cache", "tmp")

for p in [
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["TORCH_HOME"],
    os.environ["TMPDIR"],
]:
    os.makedirs(p, exist_ok=True)

# =========================================================
# 1) ERST JETZT IMPORTS
# =========================================================
import torch
import pyiqa
from PIL import Image
from torchvision.transforms import ToTensor


def resize_max_side_lanczos(img: Image.Image, max_side: int = 1024):
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS), True
    return img, False


class QAlignScorer:
    """
    Q-Align scorer via pyiqa.

    Wichtig:
    - Wenn du CUDA_VISIBLE_DEVICES=3 setzt, ist die "sichtbare" GPU intern cuda:0.
    - Tensor-Input MUSS in [0,1] sein (pyiqa validiert das).
    """
    def __init__(self, device: str = "cuda"):
        if torch.cuda.is_available():
            # "cuda" nutzt die 1 sichtbare GPU (nach CUDA_VISIBLE_DEVICES)
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        print(f"🚀 Initialisiere Q-Align auf {self.device}...")
        self.model = pyiqa.create_metric("qalign", device=self.device)
        self.to_tensor = ToTensor()

    def predict_from_pil(self, img_pil: Image.Image):
        # pyiqa erwartet [0,1] bei Tensor-Input -> KEIN *255
        img_tensor = self.to_tensor(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            aes = self.model(img_tensor, task_="aesthetic").item()
            qlt = self.model(img_tensor, task_="quality").item()
        return {"qalign_aesthetic": aes, "qalign_quality": qlt}

    def predict_from_path(self, image_path: str):
        with torch.no_grad():
            aes = self.model(image_path, task_="aesthetic").item()
            qlt = self.model(image_path, task_="quality").item()
        return {"qalign_aesthetic": aes, "qalign_quality": qlt}
