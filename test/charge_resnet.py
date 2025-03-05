import torch
import torch.nn as nn
from torchvision import models

# Charger ResNet pré-entraîné
model = models.resnet50(pretrained=True)

# Remplacer la dernière couche pour correspondre au nombre de tags dans ton dataset
num_tags = len(dataset.labels)  # Le nombre de tags (colonnes) dans ton dataset
model.fc = nn.Linear(model.fc.in_features, num_tags)

# Passer le modèle sur le GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Afficher la structure du modèle
print(model)
