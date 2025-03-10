import json
import os

import joblib
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

from build_model import build_efficientnet_model
from combined_features import EfficientNetB3Features


def run_final_pipeline():
    data = pd.read_csv("../data/images/age_detection.csv")

    train_data = data[data["split"] == "train"]
    X_train = train_data["file"]
    y_train = train_data["age"]
    
    # Load best parameters from best_params.json
    params_path = os.path.join(os.path.dirname(__file__), "best_params.json")
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("Loaded best parameters:", best_params)

    # Encoding age group labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_encoded = y_train_encoded.astype("int32")
    joblib.dump(label_encoder, "../src/label_encoder.pkl")

    final_combined_features = EfficientNetB3Features(
        image_dir="../data/images",
        target_size=(224, 224)
    )

    input_shape = (224, 224, 3)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    final_efficientnet_classifier = KerasClassifier(
        model=build_efficientnet_model,
        model__input_shape=input_shape,
        model__metrics=['accuracy'],
        model__callbacks=[early_stopping],
        dropout_rate=best_params.get("nn__dropout_rate", 0.4),
        l2_reg=0.016,
        l1_reg=0.006,
        learning_rate=best_params.get("nn__learning_rate", 0.001),
        epochs=150,
        batch_size=5,
        verbose=1,
    )

    final_pipeline = Pipeline([("features", final_combined_features), ("nn", final_efficientnet_classifier)])
    final_pipeline.fit(X_train, y_train_encoded)

    # Saving for final model deployment
    joblib.dump(final_pipeline, "../src/final_age_detection_model.pkl")
    print("Final model saved as '../src/final_age_detection_model.pkl'")


if __name__ == "__main__":
    run_final_pipeline()
