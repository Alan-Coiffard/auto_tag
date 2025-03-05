import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import f1_score
from utils import ImageDataset, transform  # Charger les utilitaires du fichier utils.py
from torch.utils.data import DataLoader
import os
from sklearn.metrics import f1_score
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

# Spécifie le chemin du fichier CSV et du dossier d'images
csv_file = 'csv/dataset.csv'  # Remplace par ton chemin vers dataset.csv
img_dir = 'dataset'  # Dossier contenant les images

# Charger ton dataset personnalisé
dataset = ImageDataset(csv_file, img_dir, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Charger le modèle ResNet pré-entraîné
model = resnet50(weights=ResNet50_Weights.DEFAULT)  # Remplacer 'pretrained=True' par 'weights=ResNet50_Weights.DEFAULT'

# Remplacer la dernière couche par une couche adaptée à ton nombre de labels
num_tags = len(dataset.labels)  # Nombre de tags (par exemple 4 pour "portrait", "autoportrait", "famille", etc.)
model.fc = nn.Linear(model.fc.in_features, num_tags)

# Passer le modèle sur le bon appareil (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fonction de perte : Binary Cross-Entropy (pour la classification multi-label)
criterion = nn.BCEWithLogitsLoss()

# Optimiseur : Adam
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10  # Nombre d'époques d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mettre le modèle en mode entraînement
    running_loss = 0.0
    
    # Boucle d'entraînement
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Passer les images dans le modèle
        outputs = model(images)
        
        # Calculer la perte
        loss = criterion(outputs, labels)
        
        # Rétropropagation et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumuler la perte
        running_loss += loss.item()
    
    # Afficher la perte pour l'époque actuelle
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

model.eval()  # Mettre le modèle en mode évaluation
all_preds = []
all_labels = []

# Sans calcul des gradients
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Passer les images dans le modèle
        outputs = model(images)
        
        # Prédictions binaires (en utilisant un seuil de 0.5)
        preds = torch.sigmoid(outputs).cpu().numpy()
        preds = (preds > 0.5).astype(int)
        
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

# Calculer le F1-score global
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
f1 = f1_score(all_labels, all_preds, average='macro')  # Calculer le F1-score macro (moyenne des étiquettes)
print(f"F1-score : {f1:.4f}")
