# TRANSFER LEARNING / FINE-TUNING | EXAMPLE B
# Reuse source hidden features for a new label task
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
scale = StandardScaler().fit(A); A, B = scale.transform(A), scale.transform(B)
source = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=3000, random_state=42).fit(A, ya)
def features(X): return np.maximum(0, X @ source.coefs_[0] + source.intercepts_[0])
head = LogisticRegression(max_iter=1000).fit(features(A), (ya == 2).astype(int))
print("New binary-task test accuracy:", head.score(features(B), (yb == 2).astype(int)))
