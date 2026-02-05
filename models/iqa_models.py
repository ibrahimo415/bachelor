import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment
from PIL import Image
from torchvision.transforms import ToTensor

class CLIPScorer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.all_prompts = (
            "quality", "brightness", "noisiness", "colorfullness",
            "sharpness", "contrast", "complexity", "natural",
            "happy", "scary", "new", "warm", "real",
            "beautiful", "lonely", "relaxing"
        )

        self.metric = CLIPImageQualityAssessment(prompts=self.all_prompts).to(self.device)
        print(f"CLIP-IQA bereit auf {self.device} (16 Dimensionen).")

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = (ToTensor()(img) * 255).to(torch.uint8).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.metric(img_tensor)

        # Ergebnisse in Dictionary packen und M4-Sicher extrahieren
        return {f"clip_{p}": float(scores[p].cpu().view(-1)[0].item()) for p in self.all_prompts}