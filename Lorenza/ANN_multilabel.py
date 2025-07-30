import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import openpyxl
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

#  Ensure reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 1️ Load dataset
df = pd.read_excel(r"C:\Users\araye\Downloads\combined_with_manual_field.xlsx")

# 2️ Define target and columns to drop
target_cols = ['iseagertestmanual', 'ismysteryguestmanual', 'isresourceoptimisimmanual']
drop_cols = [
    'idProject', 'nameProject', 'productionClass', 'testCase',
    'isResourceOptimism', 'isEagerTest', 'isMysteryGuest'
]

# Features and labels
X = df.drop(columns=target_cols + drop_cols).apply(pd.to_numeric, errors='coerce').fillna(0)
Y = df[target_cols]

# 3️ Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️ Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y.values, test_size=0.2, random_state=42
)

# 5️ Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# 6️ Define neural network model
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Output layer for 3 labels
        )
    def forward(self, x):
        return self.net(x)

model = ANN(X_train.shape[1])

# 7️ Handle imbalance with pos_weight
counts = Y_train.sum(axis=0)
total = len(Y_train)
pos_weights = torch.tensor([(total - c) / (c + 1e-5) for c in counts], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8️ Training with early stopping
epochs = 100
best_loss = float('inf')
patience, patience_counter = 10, 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    # Save best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# 9️ Load best model
model.load_state_dict(best_model_state)

# 10 Generate predictions
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)



# 11 Find optimal thresholds per label to maximize F1
optimal_thresholds = []
preds_binary = torch.zeros_like(preds)

print("\n Finding optimal thresholds:")
for i in range(len(target_cols)):
    best_f1, best_thresh = 0, 0
    y_true = Y_test[:, i]
    pred_probs = preds[:, i].numpy()
    for t in np.linspace(0.1, 0.9, 50):
        binarized = (pred_probs > t).astype(int)
        f1 = f1_score(y_true, binarized)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    optimal_thresholds.append(best_thresh)
    preds_binary[:, i] = (preds[:, i] > best_thresh).int()
    print(f"{target_cols[i]} → best threshold: {best_thresh:.2f}, F1: {best_f1:.2f}")

# 12 Print final classification report
print("\n Final classification report:")
print(classification_report(Y_test, preds_binary.numpy(), target_names=target_cols, zero_division=0))

# 1️3 Print per-label F1 scores
print("\n📊 Global F1 score by label:")
for i, col in enumerate(target_cols):
    f1 = f1_score(Y_test[:, i], preds_binary[:, i])
    print(f"{col}: F1 = {f1:.3f}")


# 1️4 Summary and visualization
print("\n Positive cases per label:")
print(Y[target_cols].sum())

counts = df[target_cols].sum()
sns.barplot(x=counts.index, y=counts.values)
plt.title("Label distribution in dataset")
plt.ylabel("Number of positive cases")
plt.show()

# 15 Confusion matrices for each label
print("\n Confusion matrices:")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(target_cols):
    cm = confusion_matrix(Y_test[:, i], preds_binary[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f"Confusion Matrix - {col}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()


