# GAUSSIAN PROCESS REGRESSION | EXAMPLE C
# Held-out GP regression error
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error
X = np.linspace(0, 4, 40)[:, None]; y = np.sin(X[:, 0])
A, B, ya, yb = train_test_split(X, y, random_state=42)
model = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01, optimizer=None).fit(A, ya)
print("Held-out MAE:", mean_absolute_error(yb, model.predict(B)))
