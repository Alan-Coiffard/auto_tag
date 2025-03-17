import os
import requests
from dotenv import load_dotenv
import json
load_dotenv()
PIXABAY_KEY = os.getenv("PIXABAY_KEY")
img_list = os.getenv("IMAGE_FILE")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE"))

def pixabay_images(query, per_page=10):
    api_key = PIXABAY_KEY  # Remplace par ta clé API
    # print(PIXABAY_KEY)
    url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo&per_page={per_page}"
    response = requests.get(url)
    try:
        json_data = response.json()
        return [photo['webformatURL'] for photo in json_data.get('hits', [])]
    except requests.exceptions.JSONDecodeError:
        print("Erreur: Impossible de décoder la réponse JSON.")
        return []


def download_images(urls, output_folder, category=""):
    os.makedirs(output_folder, exist_ok=True)
    
    for i, url in enumerate(urls):
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{output_folder}/image_{i}.jpg", "wb") as f:
                f.write(response.content)
            print(f"{category} - ✅ Image {i+1}/{len(urls)} téléchargée")
        else:
            print(f"❌ Erreur pour l'image {i+1}")

with open('./json/'+img_list+'.json', encoding='utf-8') as f:
    tags_files = json.load(f)
i = 0
for category, images_dict in tags_files.items():
    category_folder = f"dataset/{category}"
    nb_files = 0
    for path, dirs, files in os.walk(category_folder):
        nb_files = len(files)
        
    print(f"{category_folder} = {nb_files}")
    if(nb_files == 0 or nb_files != SAMPLE_SIZE):
        os.makedirs(category_folder, exist_ok=True)
        download_images(images_dict, category_folder, category)