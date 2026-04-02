import gc
import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment
from PIL import Image
from torchvision.transforms import ToTensor

class CLIPScorer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.all_prompts = (
            "quality", "brightness", "noisiness", "colorfullness",
            "sharpness", "contrast", "complexity", "natural",
            "happy", "scary", "new", "warm", "real",
            "beautiful", "lonely", "relaxing"
        )

        self.metric = CLIPImageQualityAssessment(
            prompts=self.all_prompts,
            data_range=255.0
        ).to(self.device)

        self._n = 0
        print(f"CLIP-IQA bereit auf {self.device} (16 Dimensionen).")

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        max_side = 1024
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        img_tensor = (ToTensor()(img) * 255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.metric(img_tensor)

        results = {
            f"clip_{p}": float(scores[p].detach().cpu().view(-1)[0].item())
            for p in self.all_prompts
        }

        self._n += 1
        del img_tensor

        if self._n % 100 == 0:
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return results