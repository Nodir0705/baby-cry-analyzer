import os
import numpy as np
# Import sklearn before tensorflow to avoid TLS allocation error on aarch64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from config import settings


def train_cnn():
    print("Loading preprocessed data (AUGMENTED)...")
    X = np.load(os.path.join(settings.BASE_DIR, 'model', 'X_features_augmented.npy'))
    y = np.load(os.path.join(settings.BASE_DIR, 'model', 'y_labels_augmented.npy'))

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")

    # Print per-class distribution
    print("\nPer-class sample distribution:")
    for i, cls in enumerate(le.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {cls}: {count}")
    print(f"  Total: {len(y_encoded)}\n")

    # Compute class weights to handle residual imbalance
    class_weights_arr = compute_class_weight(
        'balanced', classes=np.unique(y_encoded), y=y_encoded
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print(f"Class weights: {class_weight_dict}\n")

    # Reshape X for CNN: (samples, height, width, channels)
    X = X[..., np.newaxis]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Building CNN model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    print("Training CNN on GPU...")
    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Per-class accuracy
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nPer-class accuracy:")
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"  {cls}: {acc:.4f} ({mask.sum()} samples)")

    # Save Keras model
    keras_model_path = os.path.join(settings.BASE_DIR, 'model', 'cry_model.h5')
    model.save(keras_model_path)
    print(f"\nModel saved to {keras_model_path}")

    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(settings.BASE_DIR, 'model', 'cry_model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

    # Save label encoder for inference
    import joblib
    joblib.dump(le, os.path.join(settings.BASE_DIR, 'model', 'label_encoder.joblib'))
    print("Label encoder saved.")


if __name__ == "__main__":
    train_cnn()
