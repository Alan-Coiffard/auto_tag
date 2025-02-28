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
    # Ouvrir et lire le fichier JSON
    with open('tag_list.json') as f:
        json_file = json.load(f)


        # Boucle pour parcourir chaque clé et ses éléments
        for key, elements in json_file.items():
            tags = []

            # Construire la liste des tags
            for element in elements:
                tags.append(element)
            
            # print(f"Tags pour {key}: ", tags)

            # Préparer les entrées pour le modèle CLIP avec padding et troncature activés
            inputs = processor(text=tags, images=image, padding=True, truncation=True, return_tensors="pt")

            # Passer les entrées dans le modèle pour obtenir les logits
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

            # Calculer les probabilités
            probs = logits_per_image.softmax(dim=1)

            # Afficher les probabilités
            # print(f"{img_name} : ", probs)
            prob_list = probs.tolist()[0]
            
            # Préparer les probabilités associées à chaque tag
            prob_tags = {}
            for i, tag in enumerate(tags):
                prob_tags[tag] = prob_list[i]  # Assigner la probabilité correcte à chaque tag

            # print(prob_tags)

            # Enregistrer les résultats dans le dictionnaire `tab`
            if image_link not in tab:
                tab[image_link] = []  # Crée une liste pour cette image s'il n'y en a pas encore
            tab[image_link].append({
                "category": key,
                "probability": prob_tags
            })

    print("\nLes résultats ont été sauvegardés dans 'output_results.json'.")


# img_name = "hero_image_1.jpg"  # Utilise ici une image spécifique pour le moment
# image = Image.open(img_dir + img_name)

# get_tags_eval(image)

dir = "./images/"
print(glob.glob(dir + "*.jpg"))
list_images = glob.glob(dir + "*.jpg")

for image in list_images:
    print(image)
    get_tags_eval(image)
    
# Sauvegarder `tab` dans un fichier JSON au format voulu
with open('output_results.json', 'w') as outfile:
    json.dump(tab, outfile, indent=4)