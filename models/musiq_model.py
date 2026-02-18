import os
import torch
import pyiqa
from PIL import Image
from torchvision.transforms import ToTensor

# =========================================================
# CACHE-UMLEITUNG (Konsistent mit deinen anderen Models)
# =========================================================
BASE = "/data/stud/2026-BA-ibrahim_osman"
os.environ["TORCH_HOME"] = os.path.join(BASE, "torch_cache")

class MUSIQScorer:
    """
    MUSIQ Scorer via pyiqa.
    Unterstützt verschiedene Varianten (musiq-ava, musiq-koniq, etc.)
    """
    def __init__(self, model_type: str = "musiq-ava", device: str = "cuda"):
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        print(f"🚀 Initialisiere MUSIQ ({model_type}) auf {self.device}...")
        self.model_type = model_type
        self.model = pyiqa.create_metric(model_type, device=self.device)
        self.to_tensor = ToTensor()

        # Einmal Speicher checken
        self.print_memory()

    def print_memory(self):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 VRAM Belegung (MUSIQ): {allocated:.2f} GB")

    def predict_from_pil(self, img_pil: Image.Image):
        # pyiqa kann mit PIL oder Tensor [0,1] arbeiten
        img_tensor = self.to_tensor(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(img_tensor).item()
        return {f"{self.model_type.replace('-', '_')}": score}

    def predict_from_path(self, image_path: str):
        if not os.path.exists(image_path):
            return None
        with torch.no_grad():
            score = self.model(image_path).item()
        return {f"{self.model_type.replace('-', '_')}": score}