import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import glob

# Charger le modèle et le processeur
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tab = {}

def get_tags_eval(image_link):
    image = Image.open(image_link)
    
    # Charger le fichier JSON avec la nouvelle structure
    with open('tags.json', encoding='utf-8') as f:
        json_file = json.load(f)

    # Parcourir chaque catégorie (type, ambiance, etc.)
    for category, subcategories in json_file.items():
        for subcategory, tags in subcategories.items():
            # Préparer les entrées pour le modèle CLIP
            inputs = processor(text=tags, images=image, padding=True, truncation=True, return_tensors="pt")

            # Passer les entrées dans le modèle
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

            # Calculer les probabilités
            probs = logits_per_image.softmax(dim=1)
            prob_list = probs.tolist()[0]

            # Associer chaque tag à sa probabilité
            prob_tags = {tag: prob_list[i] for i, tag in enumerate(tags)}

            # Ajouter les résultats au dictionnaire 'tab'
            if image_link not in tab:
                tab[image_link] = []
            tab[image_link].append({
                "category": category,
                "subcategory": subcategory,
                "probability": prob_tags
            })

    print(f"Résultats pour {image_link} enregistrés.")

# Lister les images du répertoire
dir = "./images/"
dir = "./images/test/"
list_images = glob.glob(dir + "*.jpg")

for image in list_images:
    get_tags_eval(image)

# Sauvegarder les résultats dans un fichier JSON
with open('output_results.json', 'w', encoding='utf-8') as outfile:
    json.dump(tab, outfile, indent=4, ensure_ascii=False)

print("Tous les résultats ont été sauvegardés dans 'output_results.json'.")
