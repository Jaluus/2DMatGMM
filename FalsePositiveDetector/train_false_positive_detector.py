import os

import numpy as np
from joblib import dump
from scripts.loader_function import load_and_partition_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

np.random.seed(42)

FILE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_DIR, "..", "Datasets", "FalsePositiveDetectorDataset")
SAVE_DIR = os.path.join(FILE_DIR, "models")

X_train, y_train, X_test, y_test = load_and_partition_data(DATA_PATH)

classifiers = {
    "L2_logistic": LogisticRegression(
        C=1, penalty="l2", solver="saga", max_iter=10000, class_weight="balanced"
    ),
}

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, y_train)
    dump(classifier, f"{SAVE_DIR}/classifier_{name}.joblib")

    # predict on the test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(
        f"{name} | ACC: {accuracy:.2%} | REC: {recall:.2%} | PRE: {precision:.2%} | F1: {f1:.2%}"
    )
