"""Autograder tests for Lab 5B — Trees & Ensembles."""

import pytest
import sys
import os
from sklearn.metrics import recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter"))

from lab_trees import (load_and_split, build_decision_tree, build_random_forest,
                       get_feature_importances, compute_pr_auc, NUMERIC_FEATURES)


@pytest.fixture
def data():
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "starter"))
    result = load_and_split()
    assert result is not None, "load_and_split returned None"
    return result


def test_data_split(data):
    X_train, X_test, y_train, y_test = data
    total = len(X_train) + len(X_test)
    assert total > 1000
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22


def test_decision_tree_exists(data):
    X_train, X_test, y_train, y_test = data
    model = build_decision_tree(X_train, y_train)
    assert model is not None
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 5
    assert hasattr(model, "classes_")


def test_random_forest_exists(data):
    X_train, X_test, y_train, y_test = data
    model = build_random_forest(X_train, y_train)
    assert model is not None
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert hasattr(model, "classes_")


def test_feature_importances_extracted(data):
    X_train, X_test, y_train, y_test = data
    model = build_random_forest(X_train, y_train)
    assert model is not None
    importances = get_feature_importances(model, NUMERIC_FEATURES)
    assert importances is not None
    assert len(importances) == len(NUMERIC_FEATURES)
    total = sum(importances.values())
    assert abs(total - 1.0) < 0.01
    values = list(importances.values())
    assert values == sorted(values, reverse=True)


def test_balanced_recall_improvement(data):
    X_train, X_test, y_train, y_test = data
    rf_default = build_random_forest(X_train, y_train)
    rf_balanced = build_random_forest(X_train, y_train, class_weight="balanced")
    assert rf_default is not None and rf_balanced is not None
    recall_default = recall_score(y_test, rf_default.predict(X_test))
    recall_balanced = recall_score(y_test, rf_balanced.predict(X_test))
    assert recall_balanced > recall_default, \
        f"Balanced recall ({recall_balanced:.3f}) should exceed default ({recall_default:.3f})"


def test_pr_auc_values(data):
    X_train, X_test, y_train, y_test = data
    rf = build_random_forest(X_train, y_train)
    rf_balanced = build_random_forest(X_train, y_train, class_weight="balanced")
    assert rf is not None and rf_balanced is not None
    auc1 = compute_pr_auc(rf, X_test, y_test)
    auc2 = compute_pr_auc(rf_balanced, X_test, y_test)
    assert auc1 is not None and auc2 is not None
    assert 0 < auc1 <= 1, f"PR-AUC should be in (0, 1], got {auc1}"
    assert 0 < auc2 <= 1, f"PR-AUC should be in (0, 1], got {auc2}"
    assert auc2 > 0.3, f"Balanced PR-AUC should be > 0.3, got {auc2}"
