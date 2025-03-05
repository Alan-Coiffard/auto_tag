import os
import json
from unidecode import unidecode

# Chemin du dataset
dataset_dir = "dataset"

# Dictionnaire pour stocker les labels
labels = {}

# Parcours des dossiers pour récupérer les chemins et tags
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Normalisation des chemins pour éviter les accents
            img_path = os.path.relpath(os.path.join(root, file), dataset_dir)
            tags = [unidecode(tag) for tag in root.split(os.sep)[1:]]
            labels[img_path] = tags

# Sauvegarde dans labels.json
with open('./json/labels.json', 'w') as f:
    json.dump(labels, f, indent=4, ensure_ascii=False)

print(f"Labels.json généré avec {len(labels)} entrées.")
