from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


img_dir = "./images/"
img_name = "hero_image_1.jpg"

res = {}

# f = open('res.json', 'w')

with open('tag_list.json') as f:
    json_file = json.load(f)
    # Boucle pour parcourir chaque clé et ses éléments
    for key, elements in json_file.items():
        tags = []
        # print(f"{key}:")
        for element in elements:
            tags.append((element))
        
        print(tags)

        # image = Image.open(img_dir + img_name)

        # inputs = processor(text=tags, images=image, return_tensors="pt")

        # outputs = model(**inputs)
        # logits_per_image = outputs.logits_per_image
        # probs = logits_per_image.softmax(dim=1)
        # print(img_name, " : ", probs)
        # res[img_name][probs]

# f.write(res)

# image = Image.open("./images/hero_image_1.jpg")


# inputs = processor(text=["cat", "dog", "bird"], images=image, return_tensors="pt")

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image
# probs = logits_per_image.softmax(dim=1)
# print(probs)