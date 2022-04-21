from sklearn import preprocessing, model_selection, neighbors
import numpy as np
import pandas as pd

df = pd.read_csv('../Datasets/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Features
X = np.array(df.drop(['class'], 1))
# Labels
y = np.array(df['class'])

# Cross-validation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# Testing
acc = clf.score(X_test, y_test)
print(acc)

example_measures = np.array([[10, 5, 5, 4, 7, 6, 7, 2, 2], [8, 3, 4, 9, 2, 6, 7, 4, 5]])
example_measures = example_measures.reshape(len(example_measures), -1)
predict = clf.predict(example_measures)
print(predict)