import argparse
import tensorflow as tf


def build_nn_model(input_shape, units=128, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds and compiles a neural network model for classifying into 5 classes.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Activation
    from tensorflow.keras.optimizers import Adam

    print("Building NN model with input_shape:", input_shape)

    # Input layer
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

    # Output layer for 5 classes
    outputs = Dense(5, activation="softmax")(x7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnet_model(
    input_shape,
    dropout_rate=0.4,
    l2_reg=0.016,
    l1_reg=0.006,
    learning_rate=0.001,
    metrics=None,
    callbacks=None,
):
    """
    Builds and compiles a model using EfficientNetB3 as a feature extractor,
    followed by additional Dense layers for classifying into 5 classes.
    """
    if metrics is None:
        metrics = ["accuracy"]

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras import regularizers
    from tensorflow.keras.optimizers import Adam

    print("Building EfficientNetB3 model with input_shape:", input_shape)
    # Build EfficientNetB3 base model (exclude top layers)
    base_model = EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="max"
    )

    # Build the full model by stacking additional layers on top
    model = Sequential(
        [
            base_model,
            BatchNormalization(),
            Dense(
                512,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                activity_regularizer=regularizers.l1(l1_reg),
                bias_regularizer=regularizers.l1(l1_reg),
            ),
            Dropout(dropout_rate),
            Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                activity_regularizer=regularizers.l2(l1_reg),
                bias_regularizer=regularizers.l1(l1_reg),
            ),
            Dropout(dropout_rate),
            Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and display summary of a model.")
    parser.add_argument(
        "--model",
        choices=["nn", "efficientnet"],
        default="nn",
        help="Which model to build: 'nn' for the dense NN, 'efficientnet' for EfficientNetB3-based model.",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=100,
        help="Input dimension for the NN model (used only if --model nn is chosen).",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=224,
        help="Input height (used only if --model efficientnet is chosen).",
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=224,
        help="Input width (used only if --model efficientnet is chosen).",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="Number of channels (used only if --model efficientnet is chosen).",
    )
    args = parser.parse_args()

    if args.model == "nn":
        # For the dense NN model, we assume a vector input.
        input_shape = (args.input_dim,)
        model = build_nn_model(input_shape=input_shape)
    else:
        # For the EfficientNetB3 model, we assume image input.
        input_shape = (args.input_height, args.input_width, args.input_channels)
        model = build_efficientnet_model(input_shape=input_shape)

    model.summary()
