import json
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

# Définition du modèle personnalisé
class MultiLabelResNet(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelResNet, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)

    def forward(self, x):
        return self.resnet(x)

# Charger le modèle
model_name = "92"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger les poids du modèle entraîné
checkpoint = torch.load(f"../models/{model_name}.pth", map_location=device)

# Déterminer dynamiquement le nombre de classes depuis le modèle sauvegardé
num_labels = checkpoint['resnet.fc.weight'].shape[0]

# Initialiser le modèle avec le bon nombre de labels
model = MultiLabelResNet(num_labels=num_labels)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Charger les labels depuis le JSON (ajusté pour inclure toutes les sous-catégories)
with open('../json/english_tags.json', encoding='utf-8') as f:
    json_file = json.load(f)

# Créer une liste complète de labels en concaténant toutes les catégories
labels = []
category_names = list(json_file.keys())
for category, tags in json_file.items():
    labels.extend(tags)

if len(labels) != num_labels:
    print(f"⚠️ Attention : le modèle a été entraîné avec {num_labels} labels, mais le fichier JSON contient {len(labels)} labels.")
    labels = labels[:num_labels]  # Ajuster au besoin

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tab = {}

def get_tags_eval(image_link):
    image = Image.open(image_link).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Obtenir les prédictions du modèle
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Convertir en probabilités

    # Créer un dictionnaire pour chaque catégorie avec ses probabilités
    result = []
    start_idx = 0
    for category in category_names:
        category_labels = json_file[category]
        category_probabilities = {category_labels[i]: float(probs[start_idx + i]) for i in range(len(category_labels))}
        result.append({
            "category": category,
            "probability": category_probabilities
        })
        start_idx += len(category_labels)

    tab[image_link] = result
    print(f"Résultats pour {image_link} enregistrés.")

# Dossier des images
dir = "./catalogue/"
list_images = glob.glob(dir + "*.jpg")

for image in list_images:
    get_tags_eval(image)

# Sauvegarder les résultats
output_file = f'output_results_{model_name}.json'
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(tab, outfile, indent=4, ensure_ascii=False)

print(f"Tous les résultats ont été sauvegardés dans '{output_file}'.")
