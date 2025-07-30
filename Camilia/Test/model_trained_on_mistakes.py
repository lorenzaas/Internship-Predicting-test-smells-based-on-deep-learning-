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

# labels = ['EagerTest_Manual', 'AssertionRoulette_Manual']
labels = ['UnknownTest_Manual', 'MagicNumberTest_Manual']

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

# errors per smells
print("\n Errors per test smells :")
for label in labels:
    n_errors = df_result[f"Error_{label}"].sum()
    print(f"- {label}: {n_errors} errors")

# total rows to review 
nb_row_to_review = df_result[[f"Error_{label}" for label in labels]].any(axis=1).sum()
print(f"\n Total lines to review (at least one mistake): {nb_row_to_review}")


# retraining on mistakes (Attempt 2)
df_errors = df_result[df_result[[f"Error_{label}" for label in labels]].any(axis=1)]
if df_errors.empty:
    print("Nothing to retrain.")
else:
    print(f"{len(df_errors)} used to retrain the model.")
    X = df_errors[features].fillna(0).values
    y = df_errors[labels].astype(int).values

    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Retraining...")

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        new_preds = torch.round(torch.sigmoid(outputs)).int().cpu().numpy()

    df_new_preds = pd.DataFrame(new_preds, columns=[f"New_Predicted_{label}" for label in labels])
    df_result_final = pd.concat([df_errors.reset_index(drop=True), df_new_preds], axis=1)

    for label in labels:
        df_result_final[f"Corrected_Error_{label}"] = df_result_final[label] != df_result_final[f"New_Predicted_{label}"]

    df_result_final.to_excel("attempt2.xlsx", index=False)

    print("\n Recap :")
    for label in labels:
        before = df_result_final[f"Error_{label}"].sum()
        after = df_result_final[f"Corrected_Error_{label}"].sum()
        print(f"{label}: before={before}, after={after}")

# retrain on mistakes (Attempt 3) 

df_errors_remaining = df_result_final[df_result_final[[f"Corrected_Error_{label}" for label in labels]].any(axis=1)]

if df_errors_remaining.empty:
    print(" No retraining needed.")
else:
    print(f"{len(df_errors_remaining)} rows will be used for the 3rd attempt.")

    X = df_errors_remaining[features].fillna(0).values
    y = df_errors_remaining[labels].astype(int).values

    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Retraining Attempt 3...")

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        third_preds = torch.round(torch.sigmoid(outputs)).int().cpu().numpy()

    df_third_preds = pd.DataFrame(third_preds, columns=[f"New_Predicted_Att3_{label}" for label in labels])
    df_result_final3 = pd.concat([df_errors_remaining.reset_index(drop=True), df_third_preds], axis=1)

    for label in labels:
        df_result_final3[f"Corrected_Error_Att3_{label}"] = df_result_final3[label] != df_result_final3[f"New_Predicted_Att3_{label}"]

    df_result_final3.to_excel("attempt3.xlsx", index=False)

    print("\n Recap Attempt 3 :")
    for label in labels:
        before = df_result_final3[f"Corrected_Error_{label}"].sum()
        after = df_result_final3[f"Corrected_Error_Att3_{label}"].sum()
        print(f"{label}: before Attempt 3 = {before}, after Attempt 3 = {after}")

    nb_row_left = df_result_final3[[f"Corrected_Error_Att3_{label}" for label in labels]].any(axis=1).sum()
    print(f"\n Total lines still incorrect after Attempt 3: {nb_row_left}")
