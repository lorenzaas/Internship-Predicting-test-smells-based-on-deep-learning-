import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
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

# labels
# labels = ['EagerTest_Manual', 'AssertionRoulette_Manual']
labels = ['UnknownTest_Manual', 'MagicNumberTest_Manual']


# features
numerical_features = df.select_dtypes(include=['number']).columns.tolist()

features = [
    'constructor', 'line', 'cbo', 'wmc', 'rfc', 'loc', 'returnsQty',
    'variablesQty', 'parametersQty', 'methodsInvokedQty', 'methodsInvokedLocalQty',
    'methodsInvokedIndirectLocalQty', 'loopQty', 'comparisonsQty', 'tryCatchQty',
    'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
    'mathOperationsQty', 'maxNestedBlocksQty', 'anonymousClassesQty',
    'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers', 'logStatementsQty'
]


# clean the data

df_clean = df.dropna(subset=labels, how='all')
df_clean.loc[:, labels] = df_clean[labels].fillna(0)
for label in labels:
    df_clean = df_clean[df_clean[label].isin([0, 1])]

# separation of data
X = df_clean[features].fillna(0).values
y = df_clean[labels].astype(int).values

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

# dataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)


# modèle ANN

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        # self.sigmoid = nn.Sigmoid()  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # out = self.sigmoid(out)
        return out

# initialisation
input_size = X_train.shape[1]
hidden_size = 256
num_classes = 2
learning_rate = 0.001


# fix weights for unbalanced classes

train_class_counts = y_train.sum(axis=0)
total_train = len(y_train)

pos_weights = torch.tensor([
    (total_train - train_class_counts[i]) / train_class_counts[i]
    for i in range(len(labels))
], dtype=torch.float32)


model = ANN(input_size, hidden_size, num_classes)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training
num_epochs = 50
losses = []
f1_eager_list = []
f1_assertion_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)


    # Évaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        outputs_sigmoid = torch.sigmoid(outputs) 
        preds = torch.round(outputs_sigmoid).int().cpu().numpy()
        y_true = y_test.astype(int)

    report_eager = classification_report(
        y_true[:, 0], preds[:, 0], output_dict=True, zero_division=0
    )
    report_assertion = classification_report(
        y_true[:, 1], preds[:, 1], output_dict=True, zero_division=0
    )

    f1_eager = report_eager.get('1', {}).get('f1-score', 0.0)
    f1_assert = report_assertion.get('1', {}).get('f1-score', 0.0)

    f1_eager_list.append(f1_eager)
    f1_assertion_list.append(f1_assert)

    # print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | F1 label1: {f1_eager:.4f} | F1 label2: {f1_assert:.4f}")


def report(): 


# if labels == ['EagerTest_Manual', 'AssertionRoulette_Manual']: Test Smell 1 = Eager Test, Test Smell 2 = Assertion Roulette
# if labels == ['UnknownTest_Manual', 'MagicNumberTest_Manual']: Test Smell 1 = Unknown Test, Test Smell 2 = Magic Number

    print("\n=== Classification Report for Test Smell 1 ===")
    print(classification_report(y_true[:, 0], preds[:, 0], target_names=["No", "Yes"], zero_division=0))

    print("\n=== Classification Report for Test Smell 2 ===")
    print(classification_report(y_true[:, 1], preds[:, 1], target_names=["No", "Yes"], zero_division=0))

    # F1-score individuel par test smell
    f1_macro_1 = f1_score(y_true[:, 0], preds[:, 0], average='macro', zero_division=0)
    f1_micro_1 = f1_score(y_true[:, 0], preds[:, 0], average='micro', zero_division=0)

    f1_macro_2 = f1_score(y_true[:, 1], preds[:, 1], average='macro', zero_division=0)
    f1_micro_2 = f1_score(y_true[:, 1], preds[:, 1], average='micro', zero_division=0)

    print("\n=== F1-scores par test smell ===")

    print(f"Test Smell 1 -> F1 macro : {f1_macro_1:.4f} | F1 micro : {f1_micro_1:.4f}")
    print(f"Test Smell 2 -> F1 macro : {f1_macro_2:.4f} | F1 micro : {f1_micro_2:.4f}")



def confusion_matrices():

    for i in range(min(len(labels), y_true.shape[1])):
        label = labels[i]
        y_true_label = y_true[:, i]
        y_pred_label = preds[:, i]

        cm = confusion_matrix(y_true_label, y_pred_label)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix - {label}")
        plt.savefig(f"confusion_matrix_{label}.png", dpi=300)
        plt.show()


def plot_results():

    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(14, 5))

    # loss

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', color='black')
    plt.title('Loss par epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # F1-scores 
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_eager_list, label='F1 Test Smell 1', marker='o', color='blue')
    plt.plot(epochs, f1_assertion_list, label='F1 Test Smell 2', marker='o', color='green')
    plt.title('F1-score per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



# uncomment to print the report and graphs

# report()
# confusion_matrices()
# plot_results()
