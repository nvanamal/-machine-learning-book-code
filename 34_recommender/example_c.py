# RECOMMENDER SYSTEMS | EXAMPLE C
# Train factors only on observed ratings
# Dr. Nagendrababu Vanamala

import numpy as np
ratings = np.array([[5., np.nan, 1.], [4., 2., np.nan], [np.nan, 5., 2.]])
observed = np.argwhere(~np.isnan(ratings))
rng = np.random.default_rng(42)
U = rng.normal(0, 0.2, (3, 2)); V = rng.normal(0, 0.2, (3, 2))
for epoch in range(400):
    for i, j in observed:
        error = U[i] @ V[j] - ratings[i, j]
        old_u, old_v = U[i].copy(), V[j].copy()
        U[i] -= 0.01 * (error * old_v + 0.01 * old_u)
        V[j] -= 0.01 * (error * old_u + 0.01 * old_v)
print("Factor estimates:", np.round(U @ V.T, 2))
# Training fit only; reserve interactions for ranking evaluation.
