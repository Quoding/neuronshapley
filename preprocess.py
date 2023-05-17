import PIL
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import os

RAW_DIR = "imagenet_raw/"
VAL_IMAGE_FILE = "val_images.txt"


# Display an image
def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


def resize_images():
    with open(VAL_IMAGE_FILE, "r") as f:
        lines = f.readlines()

        for image_path in lines:
            image_subpath = image_path.rstrip()
            image_path = RAW_DIR + image_subpath
            image_dir, image_name = image_subpath.split("/")
            img = PIL.Image.open(image_path)
            img = img.resize(size=(299, 299))
            img = np.array(img)
            if img.shape != (299, 299, 3):
                img = np.stack((img,) * 3, axis=-1)

            img = PIL.Image.fromarray(np.uint8(img))

            save_dir = f"imagenet/{image_dir}"
            os.makedirs(save_dir, exist_ok=True)
            img.save(f"{save_dir}/{image_name}")


def rename_images():
    with open(VAL_IMAGE_FILE, "r") as f:
        lines = f.readlines()

        for image_path in lines:
            image_subpath = image_path.rstrip()
            image_path = RAW_DIR + image_subpath
            image_dir, image_name = image_subpath.split("/")
            # img = PIL.Image.open(image_path)
            save_dir = f"imagenet/{image_dir}"

            os.rename(f"{save_dir}/{image_name}.JPEG", f"{save_dir}/{image_name}")
            # img = img.resize(size=(299, 299))
            # img = np.array(img)
            # # img = preprocess_input(img)
            # img = PIL.Image.fromarray(np.uint8(img))

            # os.makedirs(save_dir, exist_ok=True)
            # img.save(f"{save_dir}/{image_name}")


resize_images()
# rename_images()
