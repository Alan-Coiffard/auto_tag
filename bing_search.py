from bing_image_downloader import downloader
import json


def download(query="", limit=100, output_dir=""):
    if query != "":
        downloader.download(
            query, 
            limit,  
            output_dir = "dataset/" + output_dir, 
            adult_filter_off=True, 
            force_replace=True, 
            timeout=60)

with open('./json/tags.json', encoding='utf-8') as f:
    json_file = json.load(f)
    
    # Récupérer les sous-catégories pour chaque clé
    for category, elements in json_file.items():
        # print(f"Catégorie: {category}")
        for subcategory, tags in elements.items():
            # print(f"  - {subcategory}: {tags}")
            # print(tags)
            for tag in tags:
                print("process : ", tag)
                output_dir = category + "/" + subcategory
                # print(output_dir)
                download(
                    query= "photography " + tag, 
                    limit=100, 
                    output_dir=output_dir)


    # # Boucle pour parcourir chaque clé et ses éléments
    # for keys, elements in json_file.items():
    #     tags = []
    #     # Construire la liste des tags
    #     # print(" ----- keys", keys)
    #     print(" ----- name", elements[0])
    #     for name, element in elements:
    #         print(name)
    #         # download(element, 100, output_dir=)
             
# downloader.download(
#     "portait", 
#     limit=100,  
#     output_dir='dataset', 
#     adult_filter_off=True, 
#     force_replace=True, 
#     timeout=60)