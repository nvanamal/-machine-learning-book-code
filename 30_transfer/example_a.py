# TRANSFER LEARNING / FINE-TUNING | EXAMPLE A
# Train and freeze a source representation
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
source = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=3000, random_state=42)
source.fit(Xs, y)
features = np.maximum(0, Xs @ source.coefs_[0] + source.intercepts_[0])
print("Frozen hidden-feature shape:", features.shape)
# This is a source-fit demonstration, not target-test performance.
