import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assurez-vous que votre modèle est sur le bon périphérique (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle sauvegardé (si nécessaire)
# model = YourModel()  # Remplacez avec le nom de votre modèle
# model.load_state_dict(torch.load('model_final.pth'))
# model.to(device)

# Mode évaluation
model.eval()  # Passer en mode évaluation

# Charger le DataLoader pour les données de test
from torch.utils.data import DataLoader
from utils import ImageDataset  # Remplacez par votre propre dataset si nécessaire

# Créer le DataLoader de test
test_dataset = ImageDataset(csv_file='csv/test.csv', img_dir='dataset', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Liste pour stocker toutes les prédictions et les véritables étiquettes
all_preds = []
all_labels = []

# Faire des prédictions sur le jeu de test
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        # Passer les images dans le modèle
        outputs = model(images)

        # Appliquer la fonction sigmoïde pour obtenir les probabilités, puis appliquer un seuil de 0.5
        preds = torch.sigmoid(outputs).cpu().numpy()
        preds = (preds > 0.5).astype(int)  # Convertir en prédictions binaires

        # Stocker les prédictions et les labels
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

# Convertir les résultats en arrays numpy
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Calculer l'accuracy par étiquette (global)
correct = 0
total = 0
for i in range(all_labels.shape[1]):  # Itérer sur chaque étiquette
    correct += (all_preds[:, i] == all_labels[:, i]).sum()  # Comparer les prédictions aux vrais labels
    total += all_labels[:, i].size(0)  # Ajouter le nombre d'exemples pour chaque étiquette

accuracy = correct / total  # Accuracy totale par étiquette
print(f"Accuracy : {accuracy:.4f}")

# Calculer le F1-score (macro)
f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1-score (macro) : {f1:.4f}")

# Visualisation : Matrice de confusion pour une étiquette donnée
# Exemple pour l'étiquette 'portrait' (index 0)
cm = confusion_matrix(all_labels[:, 0], all_preds[:, 0])  # On choisit ici la première étiquette (index 0)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Afficher la matrice de confusion sous forme de heatmap
plt.title("Matrice de confusion pour l'étiquette 'portrait'")
plt.xlabel('Prédictions')
plt.ylabel('Véritables Labels')
plt.show()

# Sauvegarder le modèle si les performances sont satisfaisantes
torch.save(model.state_dict(), 'model_final.pth')
print("Modèle sauvegardé avec succès !")
