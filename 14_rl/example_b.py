# REINFORCEMENT LEARNING | EXAMPLE B
# SARSA: an on-policy corridor learner
# Dr. Nagendrababu Vanamala

import numpy as np
rng = np.random.default_rng(42)
Q = np.zeros((5, 2)); alpha, gamma, eps = 0.2, 0.95, 0.2
def action(s):
    return int(rng.integers(2)) if rng.random() < eps else int(Q[s].argmax())
for episode in range(600):
    s = 0; a = action(s)
    for step in range(50):
        ns = int(np.clip(s + (1 if a else -1), 0, 4))
        done = ns == 4; r = 1.0 if done else -0.02
        na = action(ns)
        future = 0.0 if done else Q[ns, na]
        Q[s, a] += alpha * (r + gamma * future - Q[s, a])
        s, a = ns, na
        if done: break
print("SARSA greedy actions:", Q[:4].argmax(axis=1))
