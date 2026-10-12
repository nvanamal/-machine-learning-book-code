# UNSUPERVISED LEARNING | EXAMPLE C
# Unlabeled anomaly screening
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(42)
Xtr = rng.normal(size=(400, 2))
Xquery = np.array([[0., 0.], [6., 6.]])
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(Xtr)
print("Predictions (+1 usual, -1 unusual):", model.predict(Xquery))
print("Decision scores:", model.decision_function(Xquery))
# Unusual is not the same as a confirmed fault.
