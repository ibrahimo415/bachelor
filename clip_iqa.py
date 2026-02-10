import torch
import pyiqa

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

metric = pyiqa.create_metric("clipiqa", device=device).eval()   # oder: "clipiqa+"
score = metric("iaa_pub1_.jpg")  # Pfad als Input geht
print("pyiqa clipiqa:", float(score))
