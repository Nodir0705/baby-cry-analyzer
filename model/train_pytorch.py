"""PyTorch CNN trainer matching the TF/Keras architecture exactly.
Loads precomputed mel-spec features. Designed to run on Jetson Xavier (PyTorch 2.4.1).
Saves Keras-equivalent .pt checkpoint + label encoder + feature stats.

Usage: python3 train_pytorch.py <variant_dir>
e.g. python3 train_pytorch.py /home/jarvis/baby_cry/model/variant_B
"""
import sys, os, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

VARIANT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
print(f"Variant dir: {VARIANT_DIR}")

# ─── Load features ───
X_tr = np.load(VARIANT_DIR / "X_train.npy")
y_tr = np.load(VARIANT_DIR / "y_train.npy", allow_pickle=True)
X_va = np.load(VARIANT_DIR / "X_val.npy")
y_va = np.load(VARIANT_DIR / "y_val.npy", allow_pickle=True)
X_te = np.load(VARIANT_DIR / "X_test.npy")
y_te = np.load(VARIANT_DIR / "y_test.npy", allow_pickle=True)

le = LabelEncoder()
y_tr_e = le.fit_transform(y_tr)
y_va_e = le.transform(y_va)
y_te_e = le.transform(y_te)
num_classes = len(le.classes_)
print(f"Classes: {list(le.classes_)}")
print(f"Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")

# Standardize
mean = float(X_tr.mean()); std = float(X_tr.std())
X_tr = (X_tr - mean) / std
X_va = (X_va - mean) / std
X_te = (X_te - mean) / std
print(f"Feature stats: mean={mean:.2f} std={std:.2f}")

# Add channel dim: (N, 1, H, W)
X_tr = torch.from_numpy(X_tr).unsqueeze(1).float()
X_va = torch.from_numpy(X_va).unsqueeze(1).float()
X_te = torch.from_numpy(X_te).unsqueeze(1).float()
y_tr_t = torch.from_numpy(y_tr_e).long()
y_va_t = torch.from_numpy(y_va_e).long()
y_te_t = torch.from_numpy(y_te_e).long()


# ─── Model (matches Keras architecture exactly) ───
class CryCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 32, 3, padding=1); self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1); self.bn1b = nn.BatchNorm2d(32)
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1); self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1); self.bn2b = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.30)
        self.dropout3 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(128, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CryCNN(num_classes).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# Class weights
cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_tr_e)
class_weight_t = torch.tensor(cw, dtype=torch.float32).to(device)
print(f"Class weights: {[round(float(w),2) for w in cw]}")

criterion = nn.CrossEntropyLoss(weight=class_weight_t)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

train_loader = DataLoader(TensorDataset(X_tr, y_tr_t), batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(TensorDataset(X_va, y_va_t), batch_size=64, shuffle=False, num_workers=0)

best_val_loss = float("inf")
patience = 0
best_state = None

print("\n=== Training ===")
t_start = time.time()
for epoch in range(1, 81):
    model.train()
    tr_loss = 0; tr_correct = 0; tr_n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
        tr_correct += (logits.argmax(1) == yb).sum().item()
        tr_n += xb.size(0)
    tr_loss /= tr_n; tr_acc = tr_correct / tr_n

    model.eval()
    va_loss = 0; va_correct = 0; va_n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            va_loss += loss.item() * xb.size(0)
            va_correct += (logits.argmax(1) == yb).sum().item()
            va_n += xb.size(0)
    va_loss /= va_n; va_acc = va_correct / va_n

    print(f"Epoch {epoch:3d}  loss={tr_loss:.4f} acc={tr_acc:.3f}  val_loss={va_loss:.4f} val_acc={va_acc:.3f}  lr={optimizer.param_groups[0]['lr']:.5f}", flush=True)
    scheduler.step(va_loss)
    if va_loss < best_val_loss - 1e-4:
        best_val_loss = va_loss; patience = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience += 1
        if patience >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nTraining time: {time.time()-t_start:.0f}s")

# Restore best
if best_state is not None:
    model.load_state_dict(best_state)

# ─── Test ───
model.eval()
te_loss = 0; te_correct = 0
y_pred_all = []
with torch.no_grad():
    for i in range(0, len(X_te), 64):
        xb = X_te[i:i+64].to(device); yb = y_te_t[i:i+64].to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        te_loss += loss.item() * xb.size(0)
        te_correct += (logits.argmax(1) == yb).sum().item()
        y_pred_all.extend(logits.argmax(1).cpu().numpy().tolist())
te_loss /= len(X_te); te_acc = te_correct / len(X_te)
y_pred = np.array(y_pred_all)

print(f"\n=== TEST: loss={te_loss:.4f} acc={te_acc*100:.1f}% ===\n")
print(classification_report(y_te_e, y_pred, target_names=list(le.classes_), digits=3, zero_division=0))
print("Confusion (rows=true, cols=pred):")
cm = confusion_matrix(y_te_e, y_pred)
print(" " * 12 + "  ".join(f"{c:>10}" for c in le.classes_))
for i, c in enumerate(le.classes_):
    print(f"{c:<12}" + "  ".join(f"{cm[i,j]:>10}" for j in range(num_classes)))

# Save artifacts
out_dir = VARIANT_DIR
torch.save(model.state_dict(), out_dir / "model_pytorch.pt")
with open(out_dir / "meta.json", "w") as f:
    json.dump({
        "classes": list(le.classes_),
        "feature_mean": mean,
        "feature_std": std,
        "test_acc": float(te_acc),
        "test_loss": float(te_loss),
        "num_classes": int(num_classes),
        "n_params": int(n_params),
    }, f, indent=2)
print(f"\nSaved to {out_dir}/model_pytorch.pt + meta.json")
