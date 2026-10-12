# MULTILAYER PERCEPTRON (MLP) | EXAMPLE C
# Fit a small neural classifier
# Dr. Nagendrababu Vanamala

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(12,), solver="lbfgs", max_iter=3000, random_state=42))
model.fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte), zero_division=0))
