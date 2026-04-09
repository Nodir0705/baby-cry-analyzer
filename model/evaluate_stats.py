import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from config import settings

def evaluate_model():
    print("Loading augmented test data...")
    X = np.load(os.path.join(settings.BASE_DIR, 'model', 'X_features_augmented.npy'))
    y = np.load(os.path.join(settings.BASE_DIR, 'model', 'y_labels_augmented.npy'))
    le = joblib.load(os.path.join(settings.BASE_DIR, 'model', 'label_encoder.joblib'))
    
    # Reshape for CNN
    X = X[..., np.newaxis]
    
    # Use the same split logic as training to get the test set
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Loading TFLite model for evaluation...")
    interpreter = tf.lite.Interpreter(model_path=os.path.join(settings.BASE_DIR, 'model', 'cry_model.tflite'))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    y_pred = []
    print(f"Evaluating {len(X_test)} samples...")
    for i in range(len(X_test)):
        input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(le.classes_[np.argmax(output_data)])
        
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Calculate simple mAP equivalent (Macro Average Precision)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Macro Average Precision (mAP equivalent): {report['macro avg']['precision']:.4f}")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")

if __name__ == "__main__":
    evaluate_model()
