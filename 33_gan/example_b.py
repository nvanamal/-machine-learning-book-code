# GENERATIVE ADVERSARIAL NETWORK (GAN) | EXAMPLE B
# Train a one-dimensional location GAN
# Dr. Nagendrababu Vanamala

import numpy as np
rng = np.random.default_rng(42)
a, b, theta = 0.0, 0.0, 0.0
for step in range(2000):
    real = rng.normal(2, 1, 128); fake = rng.normal(0, 1, 128) + theta
    dr = 1 / (1 + np.exp(-np.clip(a * real + b, -30, 30)))
    df = 1 / (1 + np.exp(-np.clip(a * fake + b, -30, 30)))
    a += 0.03 * (np.mean((1 - dr) * real) - np.mean(df * fake))
    b += 0.03 * (np.mean(1 - dr) - np.mean(df))
    z = rng.normal(0, 1, 128)
    dg = 1 / (1 + np.exp(-np.clip(a * (z + theta) + b, -30, 30)))
    theta += 0.03 * a * np.mean(1 - dg)
print("Learned generator location:", round(theta, 3))
print("Target location:", 2.0)
