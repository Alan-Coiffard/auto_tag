from bing_image_downloader import downloader
import json

import sys
json_dir = ""
if(len(sys.argv) < 2):
    print("Error: please add a json file to use")
    exit(-1)
json_dir = sys.argv[1]

print (json_dir)

def download(query="", limit=100, output_dir=""):
    if query != "":
        downloader.download(
            query, 
            limit,  
            output_dir = "dataset/" + json_dir + "/" + output_dir, 
            adult_filter_off=True, 
            force_replace=True, 
            timeout=60)
        

with open('./json/'+ json_dir +'.json', encoding='utf-8') as f:
    json_file = json.load(f)
    
    # Récupérer les sous-catégories pour chaque clé
    for category, elements in json_file.items():
        # print(f"Catégorie: {category}")
        for tag in elements:
            print("process : ", tag)
            output_dir = category
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