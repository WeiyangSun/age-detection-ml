import joblib
import pandas as pd

from pipeline import grid_search, build_nn_model
from combined_features import CombinedFeatures
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def run_final_pipeline():
    data = pd.read_csv("../data/images/age_detection.csv")

    train_data = data[data["split"] == "train"]
    X_train = train_data["file"]
    y_train = train_data["age"]

    # Encoding age group labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_encoded = y_train_encoded.astype("int32")

    best_params = grid_search.best_params_

    final_combined_features = CombinedFeatures(
        image_dir="../data/images",
        target_size=(224, 224),
        augment=True,
        pca_components=best_params["features__pca_components"],
    )

    feature_sample = final_combined_features.fit_transform(X_train)
    print("Combined feature sample shape:", feature_sample.shape)
    input_shape = feature_sample.shape[1:]

    final_nn_classifier = KerasClassifier(
        model=build_nn_model,
        model__input_shape=input_shape,
        units=best_params["nn__units"],
        dropout_rate=best_params["nn__dropout_rate"],
        learning_rate=best_params["nn__learning_rate"],
        epochs=20,
        batch_size=5,
        verbose=1,
    )

    final_pipeline = Pipeline([("features", final_combined_features), ("nn", final_nn_classifier)])
    final_pipeline.fit(X_train, y_train_encoded)

    # Saving for final model deployment
    joblib.dump(final_pipeline, "../src/final_age_detection_model.pkl")
    print("Final model saved as '../src/final_age_detection_model.pkl'")

if __name__ == "__main__":
    run_final_pipeline()