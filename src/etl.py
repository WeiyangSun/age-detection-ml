import os
import random

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin


class ImageLoaderAugmentor(BaseEstimator, TransformerMixin):
    """
    Transformer to load images from a directory, apply preprocessing and
    optionally perform data augmentation.
    """

    def __init__(self, image_dir, target_size=(64, 64), augment=False):
        self.image_dir = image_dir
        self.target_size = target_size
        self.augment = augment

    def fit(self, X, y=None):
        return self

    def apply_augmentation(self, img):
        # Random Horizontal Flip
        if random.random() > 0.5:
            img = ImageOps.mirror(img)

        # Random Rortation between -15 to 15 degs
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)

        return img

    def transform(self, X, y=None):
        images = []
        for rel_path in X:
            img_path = os.path.join(self.image_dir, rel_path)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.target_size)
                if self.augment:
                    img = self.apply_augmentation(img)
                # Normalize Pixel Array
                img_arr = np.array(img) / 255
                images.append(img_arr.flatten())

        return np.array(images)
