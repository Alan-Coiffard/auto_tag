import os
for f in os.listdir("dataset"):
    print(f)
    for path, dirs, files in os.walk(f"dataset/{f}"):
        print (len(files))