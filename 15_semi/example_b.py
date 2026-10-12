# SEMI-SUPERVISED LEARNING | EXAMPLE B
# Graph-based label spreading
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
yp = ytr.copy(); yp[np.random.default_rng(42).random(len(yp)) < 0.8] = -1
model = make_pipeline(StandardScaler(),
    LabelSpreading(kernel="knn", n_neighbors=7, max_iter=100))
model.fit(Xtr, yp)
print("Labeled test accuracy:", model.score(Xte, yte))
