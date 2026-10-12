# REINFORCEMENT LEARNING — Example C

**Evaluate a fixed policy without exploration**  
Book page: **62** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/14_rl/example_c.py) · [All topics](../../README.md)

## What this example teaches

An agent observes a state, selects an action, receives a reward and next state, and improves its policy. The objective is long-term return, not merely immediate reward. Q-learning is one example, not the entire field.

## Python code

```python
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
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `policy = np.ones(4, dtype=int)` | Define the fixed policy being evaluated. |
| 3 | `returns, successes = ([], [])` | Prepare storage for episode results. |
| 4 | `for episode in range(20):` | Repeat episodes or steps of interaction. |
| 5 | `    s, total, done = (0, 0.0, False)` | Initialize or update the current state. |
| 6 | `    for step in range(50):` | Repeat episodes or steps of interaction. |
| 7 | `        a = int(policy[s])` | Choose an action by the exploration policy. |
| 8 | `        s = int(np.clip(s + (1 if a else -1), 0, 4))` | Initialize or update the current state. |
| 9 | `        done = s == 4` | Check whether the true goal is reached. |
| 10 | `        total += 1.0 if done else -0.02` | Accumulate the episode reward. |
| 11 | `        if done:` | Check the exploration or stopping condition. |
| 12 | `            break` | Stop this episode at the true terminal goal. |
| 13 | `    returns.append(total)` | Record the episode result for summarization. |
| 14 | `    successes.append(done)` | Record the episode result for summarization. |
| 15 | `print('Mean undiscounted return:', np.mean(returns))` | Print the result shown in the output box. |
| 16 | `print('Success rate:', np.mean(successes))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Mean undiscounted return: 0.9400000000000001
Success rate: 1.0
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The reported return and success rate evaluate a known fixed policy with exploration disabled.

## Try it yourself

Evaluate a deliberately bad fixed policy, then a learned policy under the same conditions.

## Important check

This output is a fixed-policy demonstration; it must not be presented as a trained-agent benchmark.
