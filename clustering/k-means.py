from matplotlib import style
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

style.use('seaborn-pastel')
plt.grid(True, linestyle='dotted')
plt.minorticks_on()

# Second two colors are the analogous
colors = ['#5cddff', '#ff6a85', '#05add9', '#ff002f']

class K_means:

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data, plot=True):
        '''
        L2 norm used as distance metric
        '''
        self.centroids = {}

        for i in range(self.k):
            '''
            Set initial centroids; better approach to randomly pick any k data points

            Clustering can succeed or fail depending on centroid initialization
            '''
            self.centroids[i] = data[i]
        
        for i in range(self.max_iter):
            '''
            Training step.

            Keys will be centroids
            '''
            # Dictionary of clusters; each cluster will have a list of points
            self.clusters = {}
            # For debugging
            # self.distances = {}

            for i in range(self.k):
                self.clusters[i] = []
                # For debugging
                # self.distances[i] = []
            
            for point in data:
                '''
                Calculate distance of each point to each centroid, store in an array
                Store closest centroid as its cluster number
                '''
                distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
                
                # First index is taken for equal distance values
                cluster = distances.index(min(distances))
                self.clusters[cluster].append(point)
                
                # For debugging
                # self.distances[cluster].append((distances, point, cluster))

            # Track
            prev_centroids = dict(self.centroids)
            
            # Update and find mean centroid point
            for cluster in self.clusters:
                self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

            optimized = True

            for c in self.centroids:
                '''
                Compare changes between centroids in each iteration, for each centroid
                '''
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if np.sum((current_centroid - original_centroid)/original_centroid * 100) > self.tol:
                    optimized = False
            
            if optimized:
                break
        
        if plot:
            # Plot centroids
            for centroid in self.centroids:
                plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
                            marker='x', color=colors[centroid + 2], s=25, linewidths=2.5)

            # Plot training data
            for cluster in self.clusters:
                color = colors[cluster]

                for point in self.clusters[cluster]:
                    plt.scatter(point[0], point[1], marker='^',
                                color=color, s=10, linewidths=1.5)
            
            plt.show()

    
    def predict(self, data):
        '''
        Test data classification

        Remarks
        Equal distance ties are broken by order
        '''
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        cluster = distances.index(min(distances))

        return cluster


    def predict_mesh(self, data, step=0.5):
        '''
        Generate a grid of 2D points, then make predictions
        '''
        X_mesh = np.arange(data[:, 0].min(), data[:, 0].max(), step)
        Y_mesh = np.arange(data[:, 1].min(), data[:, 1].max(), step)

        xx, yy = np.meshgrid(X_mesh, Y_mesh)

        # Plot grid of points to delineate decision boundary
        for point in zip(xx.ravel(), yy.ravel()):
            predictions = self.predict(point)

            plt.scatter(point[0], point[1], marker='.', color=colors[predictions])
        # plt.show()


if __name__ == '__main__':
    '''
    Use from any of the following datasets:

    1. Uniformly generated dataset of N points
    2. Cluster of points from sklearn implementation
    '''
    # Random Dataset
    # X = np.random.randint(low=-5, high=5, size=(100, 2))

    # Blob Dataset
    X, _ = make_blobs(n_samples=50, centers=2, n_features=2)

    # Show points
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    model = K_means()
    model.fit(X, plot=True)
    model.predict_mesh(X)
