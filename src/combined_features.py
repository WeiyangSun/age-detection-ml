from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from etl import ImageLoaderAugmentor
from feature_engineering import PCAFeatureExtractor, CNNFeatureExtractor


class CombinedFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that combines ETL and FeatureEngineering Scripts into a
    single feature set.
    """

    def __init__(self, image_dir, target_size=(224, 224), augment=False, pca_components=50):
        self.image_dir = image_dir
        self.target_size = target_size
        self.augment = augment
        self.pca_components = pca_components
        self._build_union()

    def _build_union(self):
        # PCA
        self.pca_pipeline = Pipeline(
            [
                (
                    "etl",
                    ImageLoaderAugmentor(
                        image_dir=self.image_dir,
                        target_size=self.target_size,
                        augment=self.augment,
                    ),
                ),
                ("pca", PCAFeatureExtractor(n_components=self.pca_components)),
            ]
        )

        # CNN
        self.cnn_pipeline = Pipeline(
            [
                (
                    "etl",
                    ImageLoaderAugmentor(
                        image_dir=self.image_dir,
                        target_size=self.target_size,
                        augment=self.augment,
                    ),
                ),
                ("cnn", CNNFeatureExtractor(target_size=self.target_size)),
            ]
        )

        # Combine
        self.union = FeatureUnion(
            [("pca_features", self.pca_pipeline), ("cnn_features", self.cnn_pipeline)]
        )

    def fit(self, X, y=None):
        self.union.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.union.transform(X)
