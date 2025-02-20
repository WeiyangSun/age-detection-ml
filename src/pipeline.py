import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from scikeras.wrappers import KerasClassifier

from combined_features import CombinedFeatures


def build_nn_model(input_shape, units=128, dropout_rate=0.2, learning_rate=0.001):

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Activation
    from tensorflow.keras.optimizers import Adam

    print("Building NN model with input_shape:", input_shape)
    # Input Layers
    inputs = Input(shape=input_shape)

    # Layer 1
    x1 = Dense(units, activation="relu")(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout_rate)(x1)

    # Layer 2
    x2 = Dense(units, activation="relu")(x1)
    x2 = BatchNormalization()(x2)

    # Layer 3
    x3 = Dense(units // 2, activation="relu")(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(dropout_rate)(x3)

    # Layer 4 - Residual Block
    res = Dense(units // 2, activation="linear")(x2)
    res = BatchNormalization()(res)
    x3 = Add()([x3, res])
    x3 = Activation("relu")(x3)

    # Layer 5
    x5 = Dense(units // 2, activation="relu")(x3)
    x5 = BatchNormalization()(x5)
    x5 = Dropout(dropout_rate)(x5)

    # Layer 6
    x6 = Dense(units // 4, activation="relu")(x5)
    x6 = BatchNormalization()(x6)

    # Layer 7
    x7 = Dense(units // 4, activation="relu")(x6)
    x7 = Dropout(dropout_rate)(x7)

    # Output Layer
    outputs = Dense(5, activation="softmax")(x7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


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

combined_features = CombinedFeatures(
    image_dir="../data/images", target_size=(224, 224), augment=True, pca_components=50
)

feature_sample = combined_features.fit_transform(X_train)
print("Combined feature sample shape:", feature_sample.shape)
input_shape = feature_sample.shape[1:]

nn_classifier = KerasClassifier(
    model=build_nn_model,
    model__input_shape=input_shape,
    units=128,
    dropout_rate=0.2,
    learning_rate=0.001,
    epochs=20,
    batch_size=5,
    verbose=1,
)

pipeline = Pipeline([("features", combined_features), ("nn", nn_classifier)])

# For Hyperparameter Tuning
param_grid = {
    "features__pca_components": [80, 100],
    "nn__units": [64],
    "nn__learning_rate": [0.001, 0.0001],
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
