import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
import pandas as pd

df = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-wisconsin.data')

# Data cleanup
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)


# Classifier
# Pass in keyword argument for kernel to change model
def classifier(kernel_):
    return svm.SVC(kernel='{}'.format(kernel_))


# Features
X = np.array(df.drop(['class'], 1))
# Labels
y = np.array(df['class'])


model = classifier('rbf')
scores = []

# k-fold
# cv = KFold(n_splits=2, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scores = cross_val_score(model, X, y, cv=5)

# for train_idx, test_idx in cv.split(X):
#     X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
#     model.fit(X_train, y_train)
#     scores.append(model.score(X_test, y_test))

# grid_param = dict(C=np.logspace(-3, 2, 6))

# clf = GridSearchCV(estimator=model, param_grid=grid_param)
# clf.fit(X, y)

# print('Best Score: ', clf.best_score_)
# print('Best C:', clf.best_estimator_.C)

# # Final model
# final_clf = clf.best_estimator_

# Testing
# acc = final_clf.score(X_test, y_test)
# print(acc)
# print(np.mean(scores))

# Simple sample to predict
example_measures = np.array([[10, 5, 5, 4, 7, 6, 7, 2, 2],
                             [1, 2, 1, 2, 3, 1, 2, 1, 2]])
example_measures = example_measures.reshape(len(example_measures), -1)

# # Predicting the test set (Validation would be good here)
predict = model.predict(example_measures)

# # Checking class labels
# print(predict)

# Reports
# print(confusion_matrix(y_test, predict))
# print(classification_report(y_test, predict))
