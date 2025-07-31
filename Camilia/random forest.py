from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


df = pd.read_excel("Result_Manual_Validation FINAL.xlsx")

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

df_clean = df.dropna(subset=labels, how='all')
df_clean.loc[:, labels] = df_clean[labels].fillna(0)
for label in labels:
    df_clean = df_clean[df_clean[label].isin([0, 1])]

X = df_clean[features].fillna(0).values
y = df_clean[labels].astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# predict
y_pred = rf_model.predict(X_test)


# classification report

def report():

    print("\n classification report: test smell 1")
    print(classification_report(y_test[:, 0], y_pred[:, 0], target_names=["No", "Yes"], zero_division=0))

    print("\n classification report: test smell 2")
    print(classification_report(y_test[:, 1], y_pred[:, 1], target_names=["No", "Yes"], zero_division=0))

        # === F1-score individuel par test smell
    f1_macro_1 = f1_score(y_test[:, 0], y_pred[:, 0], average='macro', zero_division=0)
    f1_micro_1 = f1_score(y_test[:, 0], y_pred[:, 0], average='micro', zero_division=0)

    f1_macro_2 = f1_score(y_test[:, 1], y_pred[:, 1], average='macro', zero_division=0)
    f1_micro_2 = f1_score(y_test[:, 1], y_pred[:, 1], average='micro', zero_division=0)

    print("\n F1-scores per test smell")
    print(f"Test Smell 1 -> F1 macro : {f1_macro_1:.4f} | F1 micro : {f1_micro_1:.4f}")
    print(f"Test Smell 2 -> F1 macro : {f1_macro_2:.4f} | F1 micro : {f1_micro_2:.4f}")


# confusion matrix 
def confusion_matrices():
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap='Blues')
        

        ax.set_title(f"Confusion Matrix – {label}")
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_yticklabels(['No', 'Yes'])

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, cm[row, col],
                        ha='center', va='center',
                        color='white' if cm[row, col] > cm.max() / 2 else 'black',
                        fontsize=12)

        plt.tight_layout()
        plt.savefig(f"confusion_matrix_rf_{label}_matplotlib.png")
        plt.show()



# uncomment to print the report

# report()
# confusion_matrices()
