import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



# get the file
df = pd.read_excel("Result_Manual_Validation FINAL.xlsx")

labels = ['EagerTest_Manual', 'AssertionRoulette_Manual', 'UnknownTest_Manual', 'MagicNumberTest_Manual']

group1_labels = ['EagerTest_Manual', 'AssertionRoulette_Manual']
group2_labels = ['UnknownTest_Manual', 'MagicNumberTest_Manual']

# groupe 1 
df_group1 = df[df['EagerTest_Manual'].isin([0, 1]) | df['AssertionRoulette_Manual'].isin([0, 1])].copy()
df_group1[['EagerTest_Manual', 'AssertionRoulette_Manual']] = df_group1[['EagerTest_Manual', 'AssertionRoulette_Manual']].fillna(0)

# groupe 2 
df_group2 = df[df['UnknownTest_Manual'].isin([0, 1]) | df['MagicNumberTest_Manual'].isin([0, 1])].copy()
df_group2[['UnknownTest_Manual', 'MagicNumberTest_Manual']] = df_group2[['UnknownTest_Manual', 'MagicNumberTest_Manual']].fillna(0)



#feature
numerical_features = df.select_dtypes(include=['number']).columns.tolist()

features_per_label = {
    'EagerTest_Manual': [
        'loc', 'line', 'uniqueWordsQty', 'rfc', 'methodsInvokedQty', 'variablesQty',
        'cbo', 'assignmentsQty', 'stringLiteralsQty', 'numbersQty', 'methodsInvokedLocalQty',
        'methodsInvokedIndirectLocalQty', 'tryCatchQty', 'mathOperationsQty', 'maxNestedBlocksQty'
    ],
    'AssertionRoulette_Manual': [
        'uniqueWordsQty', 'line', 'numbersQty', 'loc', 'rfc', 'variablesQty', 'cbo',
        'methodsInvokedQty', 'stringLiteralsQty', 'assignmentsQty', 'modifiers',
        'wmc', 'loopQty', 'tryCatchQty', 'mathOperationsQty'
    ],
    'UnknownTest_Manual': [
        'line', 'cbo', 'rfc', 'loc', 'variablesQty', 'methodsInvokedQty', 
        'methodsInvokedLocalQty', 'methodsInvokedIndirectLocalQty',
        'stringLiteralsQty', 'assignmentsQty','anonymousClassesQty',
        'lambdasQty', 'uniqueWordsQty', 'numbersQty','parenthesizedExpsQty'
    ],
    'MagicNumberTest_Manual': [
        'line', 'cbo', 'wmc', 'rfc', 'loc', 
        'variablesQty', 'methodsInvokedQty','loopQty', 'comparisonsQty',
        'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
        'mathOperationsQty', 'maxNestedBlocksQty', 'uniqueWordsQty'
    ]
}


for label in labels:

    print(f"\n Training for : {label}")

    features = features_per_label[label]

    # clean

    # choose group
    if label in group1_labels:
        df_clean = df_group1.copy()
    else:
        df_clean = df_group2.copy()

    X = df_clean[features].fillna(0).values
    y = df_clean[[label]].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
        def __init__(self, input_size, hidden_size):
            super(ANN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size // 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    model = ANN(X_train.shape[1], hidden_size=256)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 50
    losses = []
    f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            preds = torch.round(outputs).int().cpu().numpy()
            y_true = y_test.astype(int)

        f1 = f1_score(y_true, preds, zero_division=0)
        f1_scores.append(f1)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | F1: {f1:.4f}")



def plotAndPrint():

    # final report

    print(f"\n Final Report for {label}")
    print(classification_report(y_true, preds, target_names=["Non", "Oui"], zero_division=0))
    
    f1_macro = f1_score(y_true, preds, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, preds, average='micro', zero_division=0)
    print(f"F1-score macro: {f1_macro:.4f}")
    print(f"F1-score micro: {f1_micro:.4f}")


    # confusion matrix
    print("\n Confusion Matrix:")
   

    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {label}")
    plt.savefig(f"confusion_matrix_{label}.png", dpi=300)
    plt.show()

        
    # plot loss and F1-score
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))


    # loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, marker='o', color='black')
    plt.title(f"Loss per epoch - {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, f1_scores, marker='o', color='blue')
    plt.title(f"F1-score per epoch - {label}")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid(True)


    plt.tight_layout()
    plt.show()



# uncomment to get report and graphs

# plotAndPrint()