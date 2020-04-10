import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

df = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-wisconsin.data')

# Data cleanup
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Features
X = np.array(df.drop(['class'], 1))
# Labels
y = np.array(df['class'])

# Cross-validation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Classifier
clf = svm.SVC()
clf.fit(X_train, y_train)

# Testing
acc = clf.score(X_test, y_test)
print(acc)

# Simple sample to predict
# example_measures = np.array([[10, 5, 5, 4, 7, 6, 7, 2, 2], [8, 3, 4, 9, 2, 6, 7, 4, 5]])
# example_measures = example_measures.reshape(len(example_measures), -1)

# Predicting the test set (Validation would be good here)
predict = clf.predict(X_test)

# Checking class labels
print(predict)

# Reports
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
