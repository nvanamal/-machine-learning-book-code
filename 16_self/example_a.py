# SELF-SUPERVISED LEARNING | EXAMPLE A
# Masked-feature reconstruction
# Dr. Nagendrababu Vanamala

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
rng = np.random.default_rng(42)
z = rng.normal(size=(600, 1))
X = z * np.array([[1., 2., 3.]])
X += rng.normal(scale=0.1, size=X.shape)
Xtr, Xte = train_test_split(X, random_state=42)
# Hide the middle feature; targets come from raw X.
visible, hidden = [0, 2], 1
model = make_pipeline(StandardScaler(), Ridge(alpha=1))
model.fit(Xtr[:, visible], Xtr[:, hidden])
pred = model.predict(Xte[:, visible])
print("Hidden-feature MSE:",
      mean_squared_error(Xte[:, hidden], pred))
# No manually annotated target labels are used.
# This tests reconstruction, not downstream transfer.
