# SEMI-SUPERVISED LEARNING | EXAMPLE A
# Confidence-based self-training
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import balanced_accuracy_score
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, stratify=y, random_state=42)
rng = np.random.default_rng(42)
y_partial = ytr.copy()
y_partial[rng.random(len(ytr)) < 0.8] = -1
base = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1000))
model = SelfTrainingClassifier(base, threshold=0.9)
model.fit(Xtr, y_partial)
print(balanced_accuracy_score(yte, model.predict(Xte)))
