import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from config import settings

def train_model():
    print("Loading preprocessed data...")
    X = np.load(os.path.join(settings.BASE_DIR, 'model', 'X_features.npy'))
    y = np.load(os.path.join(settings.BASE_DIR, 'model', 'y_labels.npy'))
    
    # Flatten features for RF (Mel spec is 2D)
    X_flat = X.reshape(X.shape[0], -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    print("Training RandomForest classifier...")
    # Using RF as a baseline since TF is not available yet
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    model_save_path = os.path.join(settings.BASE_DIR, 'model', 'baby_cry_rf.joblib')
    joblib.dump(clf, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
