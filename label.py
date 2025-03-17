import os
import json
from unidecode import unidecode

# Chemin du dataset
dataset_dir = "dataset\\"

import sys
if(len(sys.argv) < 2):
    print("Error: please add a dataset directory to use")
    exit(-1)
json_dir = sys.argv[1]
dataset_dir = dataset_dir + json_dir
print (json_dir)

# Dictionnaire pour stocker les labels
labels = {}

# Parcours des dossiers pour récupérer les chemins et tags
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Normalisation des chemins pour éviter les accents
            img_path = os.path.relpath(os.path.join(root, file), dataset_dir)
            print(img_path)
            print(dataset_dir)
            tags = [unidecode(tag) for tag in root.split(os.sep)[1:]]
            path = json_dir+'\\'+img_path
            labels[path] = tags

# Sauvegarde dans labels.json
with open('./json/' + json_dir + '_labels.json', 'w') as f:
    json.dump(labels, f, indent=4, ensure_ascii=False)

print(f"Labels.json généré avec {len(labels)} entrées.")
