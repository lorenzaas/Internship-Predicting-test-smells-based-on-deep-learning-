import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



df = pd.read_excel("Result_Manual_Validation FINAL.xlsx")

# label 

# labels = ['EagerTest_Manual']
#labels = ['AssertionRoulette_Manual']
# labels= ['UnknownTest_Manual'] 
labels = ['MagicNumberTest_Manual']


#  features 

numerical_features = df.select_dtypes(include=['number']).columns.tolist()
# features = [col for col in numerical_features if col not in labels]
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


clf = RandomForestClassifier()
clf.fit(X_train, y_train[:, 0])  

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# feature list sorted 
sorted_features = [(features[i], round(importances[i], 4)) for i in indices]

# output
print(" Feature importance (sorted) for Magic Number Test:\n")
for feature, importance in sorted_features[:15]:
    print(f"- {feature}: {importance}")

