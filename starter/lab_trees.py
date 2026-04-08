"""
Module 5 Week B — Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the
Petra Telecom churn dataset with class imbalance handling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, precision_recall_curve,
                             average_precision_score, PrecisionRecallDisplay)
import matplotlib.pyplot as plt


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES]
    y = df["churned"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def build_random_forest(X_train, y_train, n_estimators=100,
                        class_weight=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight,
                                   random_state=random_state)
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    imps = dict(zip(feature_names, model.feature_importances_))
    return dict(sorted(imps.items(), key=lambda x: x[1], reverse=True))


def compute_pr_auc(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, probs)


if __name__ == "__main__":
    result = load_and_split()
    if result:
        X_train, X_test, y_train, y_test = result
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Churn rate: {y_train.mean():.2%}")

        tree = build_decision_tree(X_train, y_train)
        if tree:
            print(f"\nDecision Tree: depth={tree.get_depth()}")
            print(classification_report(y_test, tree.predict(X_test)))

        rf = build_random_forest(X_train, y_train)
        if rf:
            importances = get_feature_importances(rf, NUMERIC_FEATURES)
            if importances:
                print(f"Top 5 features: {dict(list(importances.items())[:5])}")

        rf_balanced = build_random_forest(X_train, y_train, class_weight="balanced")
        if rf_balanced:
            print("\nBalanced Random Forest:")
            print(classification_report(y_test, rf_balanced.predict(X_test)))

        if rf and rf_balanced:
            auc_default = compute_pr_auc(rf, X_test, y_test)
            auc_balanced = compute_pr_auc(rf_balanced, X_test, y_test)
            if auc_default and auc_balanced:
                print(f"PR-AUC (default): {auc_default:.3f}")
                print(f"PR-AUC (balanced): {auc_balanced:.3f}")