# REINFORCEMENT LEARNING | EXAMPLE C
# Evaluate a fixed policy without exploration
# Dr. Nagendrababu Vanamala

import numpy as np
# This is a known fixed policy, not a claimed trained result.
policy = np.ones(4, dtype=int)  # right in every nonterminal state
returns, successes = [], []
for episode in range(20):
    s, total, done = 0, 0.0, False
    for step in range(50):
        a = int(policy[s])
        s = int(np.clip(s + (1 if a else -1), 0, 4))
        done = s == 4
        total += 1.0 if done else -0.02
        if done: break
    returns.append(total); successes.append(done)
print("Mean undiscounted return:", np.mean(returns))
print("Success rate:", np.mean(successes))
# Learned policies should also be tested across conditions and seeds.
