import json

import pandas as pd

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

from build_model import build_efficientnet_model
from combined_features import EfficientNetB3Features


def run_pipeline():
    data = pd.read_csv("../data/images/age_detection.csv")

    train_data = data[data["split"] == "train"]
    X_train = train_data["file"]
    y_train = train_data["age"]

    test_data = data[data["split"] == "test"]
    X_test = test_data["file"]
    y_test = test_data["age"]

    # Encoding age group labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_encoded = y_train_encoded.astype("int32")
    y_test_encoded = y_test_encoded.astype("int32")

    efficientnet_image_loader = EfficientNetB3Features(
        image_dir="../data/images", target_size=(224, 224)
    )
    efficientnet_input_shape = (224, 224, 3)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    efficientnet_classifier = KerasClassifier(
        model=build_efficientnet_model,
        model__input_shape=efficientnet_input_shape,
        model__metrics=["accuracy"],
        model__callbacks=[early_stopping],
        dropout_rate=0.2,
        learning_rate=0.001,
        l2_reg=0.016,
        l1_reg=0.006,
        epochs=20,
        batch_size=5,
        verbose=1,
    )

    pipeline = Pipeline([("loader", efficientnet_image_loader), ("nn", efficientnet_classifier)])

    # For Hyperparameter Tuning
    param_grid = {
        "nn__learning_rate": [0.001],
        "nn__dropout_rate": [0.2, 0.3],
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=skf, scoring="accuracy", verbose=2, error_score="raise"
    )
    grid_search.fit(X_train, y_train_encoded)

    print("Best Parameters Found:")
    print(grid_search.best_params_)

    # Evaluating on Validation Set
    test_score = round(grid_search.score(X_test, y_test_encoded), 2)
    print(f"Test Set Accuracy is: {test_score}")

    with open("best_params.json", "w") as f:
        json.dump(grid_search.best_params_, f)
    print("Best parameters saved to best_params.json")


if __name__ == "__main__":
    run_pipeline()
