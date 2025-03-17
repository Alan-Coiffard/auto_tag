import json
import sys

# Définir la limite par défaut à 0.8
limit = 0.77
limit = 0.02

# Vérifier si une limite a été passée en argument
if len(sys.argv) > 1:
    try:
        verif_limit = float(sys.argv[1])
        if not (0 < verif_limit < 1):
            raise ValueError("La limite doit être un nombre entre 0 et 1.")
        limit = verif_limit
    except ValueError as e:
        print(e)
        print("Utilisation de la limite par défaut ", limit, ".")
        sys.exit(1)  # Arrête le programme en cas d'erreur

# Charger le fichier JSON avec la nouvelle structure
with open('output_results_92.json', encoding='utf-8') as f:
    json_file = json.load(f)
    
res = {}

# Parcourir chaque image et ses tags
for name, tags in json_file.items():
    info = []  # Réinitialiser pour chaque image
    for tag in tags:
        cate = tag['category']
        # subcate = tag['subcategory']
        for probability, prob_value in tag['probability'].items():
            if prob_value > limit:
                info.append({
                    "category": cate,
                    # "subcategory": subcate,
                    "tag": probability,
                    "prob": prob_value
                })
    
    # Initialiser la clé si elle n'existe pas
    if name not in res:
        res[name] = []
    
    res[name].extend(info)

# Sauvegarder les résultats dans un fichier JSON
with open('check.json', 'w', encoding='utf-8') as outfile:
    json.dump(res, outfile, indent=4, ensure_ascii=False)

print("Les résultats filtrés ont été sauvegardés dans 'check.json'.")