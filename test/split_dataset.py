from sklearn.model_selection import train_test_split
import pandas as pd  # Ajoute cette ligne pour importer pandas

# Charger l'ensemble complet de données
df = pd.read_csv('csv/dataset.csv')

# Diviser le dataset en 70% pour l'entraînement et 30% pour le test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Sauvegarder les datasets de train et test dans des fichiers CSV distincts
train_df.to_csv('csv/train.csv', index=False)
test_df.to_csv('csv/test.csv', index=False)