# SEMI-SUPERVISED LEARNING | EXAMPLE C
# Compare with a labeled-only baseline
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
X, y = make_classification(n_samples=500, n_features=8, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
mask = np.random.default_rng(42).random(len(ytr)) < 0.2
yp = np.where(mask, ytr, -1)
def base(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
plain = base().fit(Xtr[mask], ytr[mask])
semi = SelfTrainingClassifier(base(), threshold=0.9).fit(Xtr, yp)
print("Labeled only:", plain.score(Xte, yte))
print("Self-training:", semi.score(Xte, yte))
# This demonstrates comparison; semi-supervised gains are not guaranteed.
