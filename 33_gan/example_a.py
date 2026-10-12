# GENERATIVE ADVERSARIAL NETWORK (GAN) | EXAMPLE A
# Discriminator probabilities
# Dr. Nagendrababu Vanamala

import numpy as np
x = np.array([-1., 0., 1.])
a, b = 1.0, 0.0
p_real = 1 / (1 + np.exp(-(a * x + b)))
print("Discriminator probabilities:", np.round(p_real, 3))
