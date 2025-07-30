import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import numpy as np
import random

# 1. Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Load dataset
df = pd.read_excel(r"C:\Users\araye\Downloads\combined_with_manual_field.xlsx")

# 3. Targets and columns to drop
target_cols = ['iseagertestmanual', 'ismysteryguestmanual', 'isresourceoptimisimmanual']
drop_cols = [
    'idProject', 'nameProject', 'productionClass', 'testCase',
    'isResourceOptimism', 'isEagerTest', 'isMysteryGuest'
]

# 4. All numeric features
X_full = df.drop(columns=target_cols + drop_cols).apply(pd.to_numeric, errors='coerce').fillna(0)

# 5. Top 15 features from Random Forest (manual selection)
top15_features = {
    "iseagertestmanual": [
        "SimilaritiesCoefficient", "probabilityEagerTest", "fanin", "parametersQty",
        "mathOperationsQty", "variablesQty", "loopQty", "maxNestedBlocksQty", "fanout",
        "anonymousClassesQty", "methodsInvokedLocalQty", "loc", "wmc", "NMC", "tryCatchQty"
    ],
    "ismysteryguestmanual": [
        "NRF", "probabilityEagerTest", "SimilaritiesCoefficient", "fanin", "FRNC",
        "parametersQty", "loopQty", "variablesQty", "mathOperationsQty", "ERNC",
        "fanout", "anonymousClassesQty", "NRDB", "methodsInvokedLocalQty", "maxNestedBlocksQty"
    ],
    "isresourceoptimisimmanual": [
        "FRNA", "ERNA", "variablesQty", "parametersQty", "loopQty",
        "mathOperationsQty", "NRF", "FANIN", "fanout", "loc",
        "methodsInvokedLocalQty", "SimilaritiesCoefficient", "tryCatchQty", "NMC", "wmc"
    ]
}

# 6. ANN model
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
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 7. Train and evaluate function
def train_and_evaluate(X, y, label):
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32)
    model = ANN(X.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience, counter = 10, 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_probs = torch.sigmoid(model(X_test_tensor)).numpy().flatten()

    best_thresh, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 50):
        preds = (pred_probs > t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds = (pred_probs > best_thresh).astype(int)
    print(f"Label: {label} - F1 score: {best_f1:.2f} - Optimal threshold: {best_thresh:.2f}")
    print(classification_report(y_test, preds, target_names=[f"no_{label}", label], zero_division=0))
    return best_f1

# 8. Loop over each label
for label in target_cols:
    print(f"\n{'-'*30}\nLabel: {label}\n{'-'*30}")

    y = df[label].values

    print("1. Using full features:")
    f1_full = train_and_evaluate(X_full, y, label)

    print("2. Using top 15 Random Forest features:")
    selected_cols = [f for f in top15_features[label] if f in X_full.columns]
    X_top15 = X_full[selected_cols]
    f1_top = train_and_evaluate(X_top15, y, label)

    print(f"F1 full: {f1_full:.2f} | F1 top 15: {f1_top:.2f}")
