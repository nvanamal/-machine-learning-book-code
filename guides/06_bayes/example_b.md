# NAIVE BAYES — Example B

**Multinomial NB for text counts**  
Book page: **29** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/06_bayes/example_b.py) · [All topics](../../README.md)

## What this example teaches

Uses Bayes’ theorem with a conditional-independence assumption: features are independent given the class. Gaussian, Multinomial and Bernoulli variants model different feature distributions.

## Python code

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
texts = ["win cash prize", "claim free money", "meeting tomorrow",
         "project team update", "free cash offer", "team meeting notes"]
y = [1, 1, 0, 0, 1, 0]
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1))
model.fit(texts, y)
print(model.predict(["free cash", "project meeting"]))
# Tiny language demonstration; not an accuracy benchmark.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 2 | `from sklearn.feature_extraction.text import CountVectorizer` | Import CountVectorizer. |
| 3 | `from sklearn.naive_bayes import MultinomialNB` | Import MultinomialNB. |
| 4 | `texts = (` | Provide a tiny collection of example messages. |
| 5 | `    ['win cash prize', 'claim free money', 'meeting tomorrow',` | Supply arguments to the preceding operation. |
| 6 | `    'project team update', 'free cash offer',` | Supply arguments to the preceding operation. |
| 7 | `    'team meeting notes']` | Supply arguments to the preceding operation. |
| 8 | `)` | Close the preceding multi-line operation. |
| 9 | `y = [1, 1, 0, 0, 1, 0]` | Define or generate the corresponding targets. |
| 10 | `model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1))` | Chain preprocessing with the estimator. |
| 11 | `model.fit(texts, y)` | Learn parameters from the training data. |
| 12 | `print(model.predict(['free cash', 'project meeting']))` | Print the result shown in the output box. |

## Recorded output from the book

```text
[1 0]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

CountVectorizer turns messages into nonnegative word counts. MultinomialNB combines those counts with class likelihoods.

## Try it yourself

Add a message with new vocabulary and inspect which words the fitted vectorizer recognizes.

## Important check

This six-message example only demonstrates the API. It is too small for a credible spam-filter benchmark.
