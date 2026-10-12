# HYPERPARAMETER OPTIMIZATION | EXAMPLE A
# Grid search using training-only folds
# Dr. Nagendrababu Vanamala

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
search = GridSearchCV(model, {"logisticregression__C": [0.1, 1, 10]}, cv=3)
search.fit(A, ya)
print("Selected:", search.best_params_)
print("Final test accuracy:", search.score(B, yb))
