import pandas as pd
import numpy as np
import random
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Function to train and evaluate a Random Forest model
def train_rf_model(df, target_col, drop_cols, model_name="Model"):
    print(f"\n===== {model_name} =====")

    # Data preparation
    X = df.drop(columns=[target_col] + drop_cols).apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[target_col]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model training
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Feature importance (top 15)
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

    # Plot top features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='crest')
    plt.title(f"Top 15 Features – {model_name}")
    plt.tight_layout()
    plt.show()

    # Classification report
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 3. Load dataset
df = pd.read_excel(r"C:\Users\araye\Downloads\combined_with_manual_field.xlsx")

# 4. Columns to drop (common)
drop_cols_common = [
    'idProject', 'nameProject', 'productionClass', 'testCase',
    'isResourceOptimism', 'isEagerTest', 'isMysteryGuest',
    'ismysteryguestmanual', 'isresourceoptimisimmanual', 'iseagertestmanual'
]

# 5. Train models for each manual target
train_rf_model(df, target_col='ismysteryguestmanual', drop_cols=drop_cols_common, model_name="isMysteryGuest (Manual)")
train_rf_model(df, target_col='isresourceoptimisimmanual', drop_cols=drop_cols_common, model_name="isResourceOptimism (Manual)")
train_rf_model(df, target_col='iseagertestmanual', drop_cols=drop_cols_common, model_name="isEagerTest (Manual)")
