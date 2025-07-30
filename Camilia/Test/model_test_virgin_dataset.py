import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

labels = ['EagerTest_Manual', 'AssertionRoulette_Manual']
# labels = ['UnknownTest_Manual', 'MagicNumberTest_Manual']

features = [
    'constructor', 'line', 'cbo', 'wmc', 'rfc', 'loc', 'returnsQty',
    'variablesQty', 'parametersQty', 'methodsInvokedQty', 'methodsInvokedLocalQty',
    'methodsInvokedIndirectLocalQty', 'loopQty', 'comparisonsQty', 'tryCatchQty',
    'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
    'mathOperationsQty', 'maxNestedBlocksQty', 'anonymousClassesQty',
    'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers', 'logStatementsQty'
]

# loading and preparing data

df = pd.read_excel("Result_Manual_Validation FINAL.xlsx")
df_clean = df.dropna(subset=labels, how='all')
df_clean[labels] = df_clean[labels].fillna(0)
for label in labels:
    df_clean = df_clean[df_clean[label].isin([0, 1])]

X = df_clean[features].fillna(0).values
y = df_clean[labels].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# ANN model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

input_size = X_train.shape[1]
hidden_size = 256
num_classes = len(labels)

model = ANN(input_size, hidden_size, num_classes)

train_class_counts = y_train.sum(axis=0)
total_train = len(y_train)
pos_weights = torch.tensor([
    (total_train - train_class_counts[i]) / train_class_counts[i]
    for i in range(len(labels))
], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.round(torch.sigmoid(outputs)).int().cpu().numpy()
        y_true = y_test.astype(int)

    print("Training ....")



# attempt on a 10% clean data (Attempt 1) 
df_new = pd.read_excel("Virgin Dataset Test.xlsx").sample(frac=0.1, random_state=42).reset_index(drop=True)
df_new_clean = df_new.dropna(subset=features, how='any').copy()
df_new_clean[labels] = df_new_clean[labels].fillna(0)
for label in labels:
    df_new_clean = df_new_clean[df_new_clean[label].isin([0, 1])]

X_new = df_new_clean[features].fillna(0).values
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.FloatTensor(X_new_scaled)

model.eval()
with torch.no_grad():
    outputs = model(X_new_tensor) #  use of the trained model
    preds = torch.round(torch.sigmoid(outputs)).int().cpu().numpy()


df_preds = pd.DataFrame(preds, columns=[f"Predicted_{label}" for label in labels])
df_result = pd.concat([df_new_clean.reset_index(drop=True), df_preds], axis=1)

for label in labels:
    df_result[label] = df_result[label].astype(int)
    df_result[f"Predicted_{label}"] = df_result[f"Predicted_{label}"].astype(int)
    df_result[f"Error_{label}"] = df_result[label] != df_result[f"Predicted_{label}"]

df_result.to_excel("attempt1.xlsx", index=False)

print("\n ATTEMPT 1 First evaluation on a virgin dataset")

for label in labels:
    n_total = df_result[label].sum()
    n_predicted = df_result[f"Predicted_{label}"].sum()
    n_errors = df_result[f"Error_{label}"].sum()
    n_correct = len(df_result) - n_errors
    print(f"\n {label}")
    print(f"  - Total actual positives   : {n_total}")
    print(f"  - Total predicted positives: {n_predicted}")
    print(f"  - Correct predictions      : {n_correct}")
    print(f"  - Errors                   : {n_errors}")

# rows with at least one mistake

nb_row_to_review = df_result[[f"Error_{label}" for label in labels]].any(axis=1).sum()
print(f"\n Total lines to review (at least one mistake): {nb_row_to_review} / {len(df_result)}")


# retraining on the total dataset (Attempt 2) ----------------------

print(f"\n ATTEMPT 2 Retraining on the same virgin dataset ({len(df_new_clean)} rows).")

X_full = df_new_clean[features].fillna(0).values
y_full = df_new_clean[labels].astype(int).values

X_full_scaled = scaler.transform(X_full)
X_full_tensor = torch.FloatTensor(X_full_scaled)
y_full_tensor = torch.FloatTensor(y_full)

full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
full_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)

model.train()
for epoch in range(10):
    total_loss = 0
    for bx, by in full_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("retraing")

# evaluating on the same dataset
model.eval()
with torch.no_grad():
    outputs = model(X_full_tensor)
    new_preds = torch.round(torch.sigmoid(outputs)).int().cpu().numpy()

df_new_preds = pd.DataFrame(new_preds, columns=[f"New_Predicted_{label}" for label in labels])
df_result_final = pd.concat([df_new_clean.reset_index(drop=True), df_preds, df_new_preds], axis=1)

df_result_final.to_excel("attempt2_fullRetrain.xlsx", index=False)


# reporting results after retraining

for i, label in enumerate(labels):
    y_true = df_new_clean[label].values
    y_pred = df_new_preds[f"New_Predicted_{label}"].values

    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
    true_negatives = ((y_true == 0) & (y_pred == 0)).sum()

    total_positives = (y_true == 1).sum()
    predicted_positives = (y_pred == 1).sum()

    print(f"\n {label}")
    print(f"  - Actual positives        : {total_positives}")
    print(f"  - Predicted positives     : {predicted_positives}")
    print(f"  - True Positives (TP)     : {true_positives}")
    print(f"  - False Positives (FP)    : {false_positives}")
    print(f"  - False Negatives (FN)    : {false_negatives}")
    print(f"  - True Negatives (TN)     : {true_negatives}")


# error after retraining 
for label in labels:
    df_result_final[f"Error_AfterRetrain_{label}"] = (
        df_result_final[label] != df_result_final[f"New_Predicted_{label}"]
    )

nb_row_to_review_2 = df_result_final[
    [f"Error_AfterRetrain_{label}" for label in labels]
].any(axis=1).sum()

print(f"\n total lines to review after Attempt 2 (at least one mistake): {nb_row_to_review_2} / {len(df_result_final)}")

