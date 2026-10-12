# HYPERPARAMETER OPTIMIZATION | EXAMPLE C
# Validation curve for regularization
# Dr. Nagendrababu Vanamala

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
for C in [0.01, 0.1, 1, 10]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    print(C, cross_val_score(model, A, ya, cv=3).mean())
# Do not use B/yb to choose C; they remain reserved for a final test.
