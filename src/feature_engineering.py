import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class PCAFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that applies PCA on flattened image data.
    """

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X, y=None):
        return self.pca.transform(X)


class CNNFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts features using pre-trained MobileNetV2.
    """

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

        self.model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(self.target_size[0], self.target_size[1], 3),
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []
        for img_array in X:

            # Make sure the image array has shape (height, width, 3)
            if img_array.ndim != 3 or img_array.shape != (
                self.target_size[0],
                self.target_size[1],
                3,
            ):
                # If not, attempt a reshape (this is a safeguard).
                img_array = img_array.reshape(self.target_size[0], self.target_size[1], 3)

            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            feat = self.model.predict(x)
            features.append(feat.flatten())

        return np.array(features)
