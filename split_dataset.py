import pandas as pd
from sklearn.model_selection import train_test_split

# Charger le dataset complet
dataset_file = 'csv/dataset.csv'
df = pd.read_csv(dataset_file)

# Définir le pourcentage de données pour le test (par exemple, 20% pour le test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Enregistrer les fichiers
train_df.to_csv('csv/train.csv', index=False)
test_df.to_csv('csv/test.csv', index=False)

print(f"train.csv et test.csv ont été créés avec {len(train_df)} échantillons pour l'entraînement et {len(test_df)} pour le test.")
