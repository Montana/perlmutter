import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import joblib

n_samples = 10000
n_features = 20
n_classes = 2
random_state = 42
n_estimators = 100
n_jobs = -1

x, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=15,
    n_redundant=5,
    n_classes=n_classes,
    random_state=random_state
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=random_state
)

clf = RandomForestClassifier(
    n_estimators=n_estimators,
    n_jobs=n_jobs,
    random_state=random_state
)

start_time = time.time()
clf.fit(x_train, y_train)
train_time = time.time() - start_time

y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"accuracy:{acc:.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
print(f"cv:{cv_scores}")
print(f"mean_cv:{np.mean(cv_scores):.4f}")
