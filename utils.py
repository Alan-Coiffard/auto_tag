# utils.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import pandas as pd
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Accepter les images tronquées


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        :param csv_file: Chemin vers le fichier CSV contenant les chemins d'images et les labels
        :param img_dir: Dossier contenant les images
        :param transform: Transformation à appliquer sur les images (normalisation, redimensionnement, etc.)
        """
        self.data = pd.read_csv(csv_file)  # Charger le CSV
        self.img_dir = img_dir  # Dossier des images
        self.transform = transform  # Transformation
        self.labels = self.data.drop('image_path', axis=1).columns  # Extraire les labels (tags)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['image_path'])  # Construire le chemin de l'image
        
        # Vérifier si le fichier existe avant d'essayer de l'ouvrir
        if not os.path.exists(img_path):
            print(f"Image non trouvée : {img_path}")
            return None, None

        try:
            image = Image.open(img_path).convert("RGBA")  # Ouvrir l'image
        except OSError:
            print(f"Image corrompue ou tronquée, ignorée : {img_path}")
            return None, None  # Ignorer cette image

        labels = torch.tensor(self.data.iloc[idx][1:].values.astype(float))  # Convertir les tags en tensor
        if self.transform:
            image = self.transform(image)  # Appliquer les transformations (comme le redimensionnement)
        return image, labels

# Transformation des images : Redimensionner, convertir en tensor et normaliser
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner l'image
    transforms.ToTensor(),  # Convertir l'image en tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normaliser
])

# Créer les DataLoader pour l'entraînement et le test
train_dataset = ImageDataset(csv_file='csv/train.csv', img_dir='dataset', transform=transform)
test_dataset = ImageDataset(csv_file='csv/test.csv', img_dir='dataset', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)