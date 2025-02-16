import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


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

    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []
        for rel_path in X:
            img_path = rel_path  # Adjust pathing later
            img = keras_image.load_img(img_path, target_size=self.target_size)
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = self.model.predict(x)
            features.append(feat.flatten())

        return np.array(features)
