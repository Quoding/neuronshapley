import os

out_dir = "imagenet"

with open("val_images.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        d, fn = line.split("/")
        os.makedirs(f"{out_dir}/{d}/", exist_ok=True)
        os.rename(f"{out_dir}/{fn}", f"{out_dir}/{d}/{fn}")
