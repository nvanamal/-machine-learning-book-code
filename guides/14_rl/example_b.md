# REINFORCEMENT LEARNING — Example B

**SARSA: an on-policy corridor learner**  
Book page: **61** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/14_rl/example_b.py) · [All topics](../../README.md)

## What this example teaches

An agent observes a state, selects an action, receives a reward and next state, and improves its policy. The objective is long-term return, not merely immediate reward. Q-learning is one example, not the entire field.

## Python code

```python
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
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 3 | `Q = np.zeros((5, 2))` | Initialize a state-by-action value table. |
| 4 | `alpha, gamma, eps = (0.2, 0.95, 0.2)` | Set update, discount and exploration rates. |
| 5 | `` |  |
| 6 | `def action(s):` | Define a reusable helper function. |
| 7 | `    return (` | Return the helper result to its caller. |
| 8 | `        int(rng.integers(2)) if rng.random() < eps else int(Q[s]` | Supply arguments to the preceding operation. |
| 9 | `        .argmax())` | Supply arguments to the preceding operation. |
| 10 | `    )` | Close the preceding multi-line operation. |
| 11 | `for episode in range(600):` | Repeat episodes or steps of interaction. |
| 12 | `    s = 0` | Initialize or update the current state. |
| 13 | `    a = action(s)` | Choose an action by the exploration policy. |
| 14 | `    for step in range(50):` | Repeat episodes or steps of interaction. |
| 15 | `        ns = int(np.clip(s + (1 if a else -1), 0, 4))` | Apply the action and obtain the next state. |
| 16 | `        done = ns == 4` | Check whether the true goal is reached. |
| 17 | `        r = 1.0 if done else -0.02` | Compute the reset gate. |
| 18 | `        na = action(ns)` | Choose an action by the exploration policy. |
| 19 | `        future = 0.0 if done else Q[ns, na]` | Use zero future value after true termination. |
| 20 | `        Q[s, a] += alpha * (r + gamma * future - Q[s, a])` | Update the chosen state-action value. |
| 21 | `        s, a = (ns, na)` | Advance to the next state and action. |
| 22 | `        if done:` | Check the exploration or stopping condition. |
| 23 | `            break` | Stop this episode at the true terminal goal. |
| 24 | `print('SARSA greedy actions:', Q[:4].argmax(axis=1))` | Print the result shown in the output box. |

## Recorded output from the book

```text
SARSA greedy actions: [1 1 1 1]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

SARSA uses the next action actually selected by its exploratory policy in the update target.

## Try it yourself

Compare its policy with Q-learning under the same corridor, reward and episode budget.

## Important check

SARSA is on-policy; replacing its next action value by max Q changes the algorithm.
