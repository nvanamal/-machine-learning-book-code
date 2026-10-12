# REINFORCEMENT LEARNING | EXAMPLE A
# Q-learning in a five-state corridor
# Dr. Nagendrababu Vanamala

import numpy as np
rng = np.random.default_rng(42)
Q = np.zeros((5, 2))  # corridor: states 0..4
alpha, gamma, epsilon = 0.2, 0.95, 0.2
for episode in range(600):
    s = 0  # goal is state 4; actions: left/right
    for step in range(50):
        if rng.random() < epsilon:
            a = int(rng.integers(2))
        else:
            a = int(np.argmax(Q[s]))
        ns = int(np.clip(s + (1 if a else -1), 0, 4))
        done = ns == 4
        reward = 1.0 if done else -0.02
        future = 0.0 if done else Q[ns].max()
        Q[s, a] += alpha * (
            reward + gamma * future - Q[s, a])
        s = ns
        if done:
            break
print("Greedy actions:", Q[:4].argmax(axis=1))
# True terminal goal: no bootstrap after termination.
# The 50-step cap truncates training, not the task.
