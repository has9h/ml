import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np 
style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # {||w||: [w, b]}
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]
                      
        all_data = []

    def predict(self, features):
        # sign(x.weights + bias)
        classification = np.sign(np.dot(np.array(features), self.weights) + self.bias)
        return classification


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),
             1: np.array([[5, 1],
                         [6, -1],
                         [7, 3], ])}
