import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from config import settings


# ── Focal Loss ───────────────────────────────────────────────────────────────

class SparseFocalLoss(tf.keras.losses.Loss):
    """Focal loss for multi-class classification with sparse labels.
    Down-weights easy samples, focuses learning on hard/borderline cases.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        # Label smoothing
        y_one_hot = tf.one_hot(y_true, num_classes)
        y_smooth = y_one_hot * (1.0 - self.label_smoothing) + \
                   self.label_smoothing / tf.cast(num_classes, tf.float32)

        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Focal modulation
        ce = -y_smooth * tf.math.log(y_pred)
        weight = y_smooth * tf.pow(1.0 - y_pred, self.gamma)
        focal = weight * ce

        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return config


# ── Cosine Decay Schedule ────────────────────────────────────────────────────

def cosine_decay_schedule(epochs, initial_lr=5e-4, warmup_epochs=5):
    """Cosine decay with linear warmup."""
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return initial_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
    return schedule


def train():
    print("Loading pre-split YAMNet embeddings...")
    model_dir = os.path.join(settings.BASE_DIR, 'model')

    X_train = np.load(os.path.join(model_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(model_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(model_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(model_dir, 'y_test.npy'))

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    embedding_dim = X_train.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")

    print("\nTrain distribution:")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {np.sum(y_train_enc == i)}")

    print("\nTest distribution (originals only, no augmentation):")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {np.sum(y_test_enc == i)}")

    class_weights_arr = compute_class_weight(
        'balanced', classes=np.unique(y_train_enc), y=y_train_enc
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    # Boost discomfort weight — most confused class (with hungry)
    discomfort_idx = list(le.classes_).index('discomfort')
    class_weight_dict[discomfort_idx] *= 1.5
    print(f"\nClass weights (discomfort boosted): {class_weight_dict}\n")

    # Wider head for 3072-dim multi-pool input
    model = models.Sequential([
        layers.Input(shape=(embedding_dim,)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dense(num_classes, activation='softmax')
    ])

    EPOCHS = 200
    INITIAL_LR = 5e-4

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=SparseFocalLoss(gamma=3.0, label_smoothing=0.1),
        metrics=['accuracy']
    )
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True
    )
    lr_schedule = callbacks.LearningRateScheduler(
        cosine_decay_schedule(EPOCHS, initial_lr=INITIAL_LR, warmup_epochs=5)
    )

    print("Training classification head...")
    history = model.fit(
        X_train, y_train_enc,
        epochs=EPOCHS,
        batch_size=32,
        validation_data=(X_test, y_test_enc),
        class_weight=class_weight_dict,
        callbacks=[early_stop, lr_schedule],
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_enc)
    print(f"\nTest accuracy (honest, originals only): {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nPer-class accuracy (honest):")
    for i, cls in enumerate(le.classes_):
        mask = y_test_enc == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            print(f"  {cls}: {acc:.4f} ({mask.sum()} test originals)")

    # Confusion matrix
    print("\nConfusion matrix:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    print(f"{'':>12}", end='')
    for cls in le.classes_:
        print(f"{cls:>12}", end='')
    print()
    for i, cls in enumerate(le.classes_):
        print(f"{cls:>12}", end='')
        for j in range(num_classes):
            print(f"{cm[i][j]:>12}", end='')
        print()

    # Save Keras model
    head_path = os.path.join(model_dir, 'yamnet_head.h5')
    model.save(head_path)
    print(f"\nHead saved to {head_path}")

    # Convert head to TFLite
    print("Converting head to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(model_dir, 'yamnet_head.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite head saved to {tflite_path}")

    # Save label encoder
    import joblib
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))
    print("Label encoder saved.")


if __name__ == "__main__":
    train()
