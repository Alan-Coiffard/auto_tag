from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def cls():
    os.system('cls' if os.name=='nt' else 'clear')



# Définition du Dataset personnalisé
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        try:
            image = Image.open(img_name).convert("RGB")  # Convertir en RVB
        except (OSError, Image.DecompressionBombError) as e:
            print(f"Erreur avec l'image {img_name} : {e}")
            return torch.zeros(3, 224, 224), torch.zeros(len(self.data.columns) - 1)  # image vide et labels nuls

        labels = self.data.iloc[idx, 1:].values
        labels = [float(label) if str(label).replace('.', '', 1).isdigit() else 0.0 for label in labels]
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels



# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Créer les DataLoader pour l'entraînement et le test
train_dataset = ImageDataset(csv_file='csv/train.csv', img_dir='dataset', transform=transform)
test_dataset = ImageDataset(csv_file='csv/test.csv', img_dir='dataset', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# Définition du modèle avec ResNet pré-entrainé
class MultiLabelResNet(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelResNet, self).__init__()
        # Charger le modèle ResNet pré-entrainé
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remplacer la couche fully connected pour avoir une sortie pour chaque étiquette
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)

    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialiser le modèle avec le nombre d'étiquettes que tu as (par exemple 20)
num_labels = len(train_dataset.data.columns) - 1  # Supposons que la première colonne soit le nom de l'image
model = MultiLabelResNet(num_labels=num_labels)
model = model.to(device)  # Assurez-vous que le modèle est sur le bon appareil (GPU ou CPU)


# Définir l'optimizer et la fonction de perte
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy pour la classification multi-label

# Scheduler pour ajuster le taux d'apprentissage
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Fonction pour entraîner le modèle
def train_model(model, train_dataloader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Mettre le modèle en mode entraînement
        running_loss = 0.0
        i = 0
        for inputs, labels in train_dataloader:
            cls()
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print("Progression : ", i, "/", len(train_dataloader))
            # Now, to clear the screen
            i = i+1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Remettre les gradients à zéro

            # Passer les images dans le modèle
            outputs = model(inputs)

            # Calculer la perte
            loss = criterion(outputs, labels)
            loss.backward()  # Calculer les gradients

            optimizer.step()  # Mettre à jour les poids

            running_loss += loss.item()

        # Afficher la perte après chaque époque
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")
        
        # Mettre à jour le taux d'apprentissage
        scheduler.step()

    print("Entraînement terminé !")

# Entraîner le modèle
train_model(model, train_dataloader, criterion, optimizer, scheduler, num_epochs=20)


from sklearn.metrics import f1_score
import numpy as np

# Fonction pour tester le modèle
def test_model(model, test_dataloader):
    model.eval()  # Passer en mode évaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            # Passer les images dans le modèle
            outputs = model(images)

            # Appliquer la fonction sigmoïde pour avoir des probabilités
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(int)  # Seuil de 0.5 pour classer

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    # Convertir en arrays numpy
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculer le F1-score macro
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"F1-score : {f1:.4f}")

    return f1

# Tester le modèle
f1 = test_model(model, test_dataloader)
name = input("Chose a name if you want to save : ")
if name != '':
    torch.save(model.state_dict(), 'models/' + name + '.pth')  # Sauvegarder les poids du modèle
    print("Modèle sauvegardé avec succès !")