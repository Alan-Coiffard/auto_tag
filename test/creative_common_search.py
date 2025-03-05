import requests
import json
import os

def download_images(query="", limit=30, output_dir=""):
    if query:
        url = f"https://api.creativecommons.engineering/v1/images?q={query}&page_size={limit}"
        response = requests.get(url)
        
        # Vérifier le statut de la réponse
        if response.status_code != 200:
            print(f"Erreur {response.status_code}: Impossible de récupérer les données pour '{query}'.")
            print("Réponse brute:", response.text)
            return
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"Erreur: réponse invalide pour '{query}'.")
            print("Réponse brute:", response.text)
            return
        
        # Créer le dossier de sortie
        output_path = os.path.join("dataset", output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Télécharger chaque image
        for i, item in enumerate(data.get('results', [])[:limit]):
            image_url = item.get('url')
            if not image_url:
                continue

            image_title = item.get('title', 'untitled').replace(" ", "_").replace("/", "_")
            img_data = requests.get(image_url).content
            with open(f"{output_path}/{image_title}_{i+1}.jpg", 'wb') as handler:
                handler.write(img_data)
            print(f"Téléchargé: {image_title}")

with open('tags.json', encoding='utf-8') as f:
    json_file = json.load(f)
    
    # Récupérer les sous-catégories pour chaque clé
    for category, elements in json_file.items():
        for subcategory, tags in elements.items():
            for tag in tags:
                print("Processing:", tag)
                output_dir = os.path.join(category, subcategory, tag)
                download_images(query=tag, limit=30, output_dir=output_dir)
