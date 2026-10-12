# SELF-SUPERVISED LEARNING | EXAMPLE B
# Denoising reconstruction without manual labels
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
rng = np.random.default_rng(42)
z = rng.normal(size=(500, 1))
X = z * np.array([[1., 2., 3.]]) + rng.normal(0, 0.05, (500, 3))
Xtr, Xte = train_test_split(X, random_state=42)
noisy_tr = Xtr + rng.normal(0, 0.3, Xtr.shape)
noisy_te = Xte + rng.normal(0, 0.3, Xte.shape)
model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(noisy_tr, Xtr)
print("Noisy MSE:", mean_squared_error(Xte, noisy_te))
print("Denoised MSE:", mean_squared_error(Xte, model.predict(noisy_te)))
