import pandas as pd
import json

# Charger le fichier labels.json
with open('./json/labels.json', 'r') as f:
    data = json.load(f)

# Préparer les données pour le DataFrame
rows = []
for img_path, tags in data.items():
    row = {'image_path': img_path}
    for tag in tags:
        row[tag] = 1
    rows.append(row)

# Créer le DataFrame
df = pd.DataFrame(rows).fillna(0)

# Sauvegarder au format CSV
df.to_csv('csv/dataset.csv', index=False)

print("Dataset CSV créé : csv/dataset.csv")
print(df.head())
