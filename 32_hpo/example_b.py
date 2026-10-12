# HYPERPARAMETER OPTIMIZATION | EXAMPLE B
# Random search over tree settings
# Dr. Nagendrababu Vanamala

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
    {"max_depth": [2, 3, 5, None], "min_samples_leaf": [1, 2, 4]},
    n_iter=5, cv=3, random_state=42)
search.fit(A, ya)
print("Selected:", search.best_params_)
print("Final test accuracy:", search.score(B, yb))
