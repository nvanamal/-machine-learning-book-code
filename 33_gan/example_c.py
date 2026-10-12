# GENERATIVE ADVERSARIAL NETWORK (GAN) | EXAMPLE C
# Generator loss differs from original minimax loss
# Dr. Nagendrababu Vanamala

import numpy as np
p_fake = np.array([0.1, 0.4, 0.8])
minimax = np.mean(np.log(1 - p_fake))
non_saturating = -np.mean(np.log(p_fake))
print("Minimax generator loss:", round(minimax, 4))
print("Non-saturating loss:", round(non_saturating, 4))
# Different objectives/gradient behaviour; values are not interchangeable.
