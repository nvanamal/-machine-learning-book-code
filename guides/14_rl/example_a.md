# REINFORCEMENT LEARNING — Example A

**Q-learning in a five-state corridor**  
Book page: **60** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/14_rl/example_a.py) · [All topics](../../README.md)

## What this example teaches

An agent observes a state, selects an action, receives a reward and next state, and improves its policy. The objective is long-term return, not merely immediate reward. Q-learning is one example, not the entire field.

## Python code

```python
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
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 3 | `Q = np.zeros((5, 2))` | Initialize a state-by-action value table. |
| 4 | `alpha, gamma, epsilon = (0.2, 0.95, 0.2)` | Set update, discount and exploration rates. |
| 5 | `for episode in range(600):` | Repeat episodes or steps of interaction. |
| 6 | `    s = 0` | Initialize or update the current state. |
| 7 | `    for step in range(50):` | Repeat episodes or steps of interaction. |
| 8 | `        if rng.random() < epsilon:` | Check the exploration or stopping condition. |
| 9 | `            a = int(rng.integers(2))` | Choose an action by the exploration policy. |
| 10 | `        else:` | Otherwise use the greedy learned action. |
| 11 | `            a = int(np.argmax(Q[s]))` | Choose an action by the exploration policy. |
| 12 | `        ns = int(np.clip(s + (1 if a else -1), 0, 4))` | Apply the action and obtain the next state. |
| 13 | `        done = ns == 4` | Check whether the true goal is reached. |
| 14 | `        reward = 1.0 if done else -0.02` | Assign the reward for this transition. |
| 15 | `        future = 0.0 if done else Q[ns].max()` | Use zero future value after true termination. |
| 16 | `        Q[s, a] += alpha * (reward + gamma * future - Q[s, a])` | Update the chosen state-action value. |
| 17 | `        s = ns` | Initialize or update the current state. |
| 18 | `        if done:` | Check the exploration or stopping condition. |
| 19 | `            break` | Stop this episode at the true terminal goal. |
| 20 | `print('Greedy actions:', Q[:4].argmax(axis=1))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Greedy actions: [1 1 1 1]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The greedy action IDs are 0=left and 1=right. The goal’s future value is set to zero at true termination.

## Try it yourself

Change exploration epsilon and compare success across several training seeds.

## Important check

Training return and a learned policy alone are insufficient for deployment safety.
