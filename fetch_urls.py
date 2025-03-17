import os
import requests
from dotenv import load_dotenv
import json

# Charger les variables d'environnement
load_dotenv()
PIXABAY_KEY = os.getenv("PIXABAY_KEY")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE"))
img_list = os.getenv("IMAGE_FILE")

def round_up(num, den):
    return -(-num // den)

replicate = 1
if SAMPLE_SIZE > 200:
    replicate = round_up(SAMPLE_SIZE, 200)
    SAMPLE_SIZE = 200

import time

def pixabay_images(query, per_page=10):
    api_key = PIXABAY_KEY
    url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo&per_page={per_page}"
    
    response = requests.get(url)
    try:
        json_data = response.json()
        if 'detail' in json_data and 'throttled' in json_data['detail'].lower():
            print("API rate limit exceeded. Waiting before retrying...")
            time.sleep(10)  # Attente de 10 secondes avant de réessayer
            return pixabay_images(query, per_page)  # Réessayer après l'attente
        return [photo['webformatURL'] for photo in json_data.get('hits', [])]
    except requests.exceptions.JSONDecodeError:
        print("Erreur: Impossible de décoder la réponse JSON.")
        return []

with open('./json/english_tags.json', encoding='utf-8') as f:
    tags_files = json.load(f)

res = {}
for category, elements in tags_files.items():
    for tag in elements:
        print("Processing :", tag)
        res[tag] = []
        for _ in range(replicate):
            res[tag].extend(pixabay_images(tag, SAMPLE_SIZE))
        print(f"NB links : {len(res[tag])}")

# Enregistrement du JSON fusionné
with open(f'./json/{img_list}.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
# import os
# import requests
# from dotenv import load_dotenv
# import json

# # Charger les variables d'environnement
# load_dotenv()
# PIXABAY_KEY = os.getenv("PIXABAY_KEY")
# SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE"))
# img_list = os.getenv("IMAGE_FILE")

# def round_up(num, den):
#     return -(-num // den)

# replicate = 1
# if SAMPLE_SIZE > 200:
#     replicate = round_up(SAMPLE_SIZE, 200)
#     SAMPLE_SIZE = 200

# def pixabay_images(query, per_page=10):
#     api_key = PIXABAY_KEY
#     url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo&per_page={per_page}"
    
#     response = requests.get(url)
#     try:
#         json_data = response.json()
#         return [photo['webformatURL'] for photo in json_data.get('hits', [])]
#     except requests.exceptions.JSONDecodeError:
#         print("Erreur: Impossible de décoder la réponse JSON.")
#         return []

# with open('./json/english_tags.json', encoding='utf-8') as f:
#     tags_files = json.load(f)

# res = {}
# for category, elements in tags_files.items():
#     for tag in elements:
#         print("Processing :", tag)
#         res[tag] = []
#         for _ in range(replicate):
#             res[tag].extend(pixabay_images(tag, SAMPLE_SIZE))
#         print(f"NB link : {len(res)}")

# # Enregistrement du JSON fusionné
# with open(f'./json/{img_list}.json', 'w', encoding='utf-8') as f:
#     json.dump(res, f, indent=4, ensure_ascii=False)
