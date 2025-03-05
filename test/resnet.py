import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights

# Charger le modèle pré-entrainé
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Préparer l'image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("./images/hero_image_1.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Prédiction
with torch.no_grad():
    out = model(batch_t)
    print(out.argmax(dim=1))
