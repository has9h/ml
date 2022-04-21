from sklearn.model_selection import KFold, cross_val_score, \
    cross_val_predict, train_test_split
from sklearn import datasets, linear_model, metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def example():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])

    # Define the split - into 2 folds
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)      # returns the number of splitting iterations in the cross-validator

    print(kf)

    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)


# Load the Diabetes dataset
columns = 'age sex bmi map_ tc ldl hdl tch ltg glu'.split()

diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

# Training and Testing
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# Model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

# The line / model
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.scatter(y_test, predictions, marker='.')

# Perform 6-fold cross validation
scores = cross_val_score(model, df, y, cv=6)
print('Cross-validated scores:', scores)

accuracy = metrics.r2_score(y, predictions)
print('Cross-Predicted Accuracy:', accuracy)

# Make cross validated predictions
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y, predictions, marker='.')
plt.show()
