# GAUSSIAN PROCESS REGRESSION | EXAMPLE A
# GP mean and standard deviation
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
X = np.linspace(0, 3, 12)[:, None]; y = np.sin(X[:, 0])
model = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01, optimizer=None)
model.fit(X, y)
mean, std = model.predict(np.array([[1.5], [4.0]]), return_std=True)
print("Means:", np.round(mean, 3))
print("Model standard deviations:", np.round(std, 3))
