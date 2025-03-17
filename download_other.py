from oidv6 import downloader
import os
import json

with open('./json/env.json', encoding='utf-8') as f:
    env = json.load(f)
    

output_dir = env['dataset_path']
os.makedirs(output_dir, exist_ok=True)

print(env['tags_file_path'])
with open(env['tags_file_path'], encoding='utf-8') as f:
    tags = json.load(f)
    
# print(tags)
for category, elements in tags.items():
    # print(f"Catégorie: {category}")
    for tag in elements:
        print("process : ", tag)
        category_dir = os.path.join(output_dir, category, tag)
        os.makedirs(category_dir, exist_ok=True)
        # openimages.download(f"{tag}", output_dir=category_dir, max_images=500)
        downloader.download_labels(
            labels=[f"{tag}"],  
            limit=500,  # Nombre d'images à télécharger
            dataset_dir=category_dir
        )
# for category in categories:
#     category_dir = os.path.join(output_dir, category)
#     os.makedirs(category_dir, exist_ok=True)
#     openimages.download(f"{category}", output_dir=category_dir, max_images=500)
