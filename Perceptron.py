"""
COMP575 - Coursework (Perceptron)
---------------------------------
By Thomas Rex Greenway 201198319

Implementation of a simple Perceptron.
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt


######### Perceptron Class ##############

class Perceptron():
    """
    Perceptron Class.

    Parameters
    ----------
    epochs : int
        Number of steps across which to train the perceptron.
    eta : float in (0, 1], Default = 0.01
        Learning rate for the perceptron.
    """
    def __init__(self, eta=0.01):
        """
        Initialises a Perceptron object with given learning rate.
        """
        self.eta = eta
        self.w = []

    def fit(self, X, y, epochs):
        """
        Trains the Perceptron (updates the weights), given a set of inputs
        with known binary class labels, -1 or 1, over desired number of epochs.
        """
        self.w = np.zeros(1 + X.shape[1])
        print(f"Initial Weights: {self.w}")

        # Add bias = 1 at index 0 for each data point
        X = np.insert(X, 0, [1], axis = 1)

        for i in range(epochs):
            # Update weights for each point in training set.
            for xi, target in zip(X, y):
                a = np.dot(xi, self.w[:])
                d = np.sign(a)
                update = self.eta * (target - d)
                self.w[:] += update * xi
            print(f"Iteration: {i} --> Weights: {self.w}")

    def predict(self, X):
        """
        Given an input array of data points, X, returns a corresponding 
        array of class predictions, -1 or 1.
        """
        if len(self.w) == 0:
            print("PLEASE TRAIN NETWORK FIRST")
            return
        # Add additional input for bias term.
        X = np.insert(X, 0, [1], axis = 1)
        return np.where(np.dot(X, self.w[:]) >= 0.0, 1, -1)



############### GET DATA ################

def iris_data(epochs = 10, eta = 0.1):
    # Get Iris Dataset (Only first two classes / 100 data points: Iris-setosa and Iris-versicolor)
    f = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    convertFunc = lambda name: 1. if name == b"Iris-setosa" else (-1 if name == b"Iris-versicolor" else 3.)
    sorted_data = np.genfromtxt(f, delimiter=",", converters={-1 : convertFunc})[:100]
    # Remove first value (for 3d plot purposes) + Permute Dataset
    sorted_data = np.delete(sorted_data, 0, 1)
    data = np.random.default_rng().permutation(sorted_data)

    # Sperate input and targets
    X = data[:, :-1]
    y = data[:, -1]

    # Create Perceptron + TRAIN
    p = Perceptron(eta)
    p.fit(X, y, epochs)

    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    w = p.w
    xx, yy = np.arange(1, 5, 0.25), np.arange(1, 5, 0.25)
    xx, yy = np.meshgrid(xx, yy)
    zz = (- w[0] - (w[1] * xx) + (w[1] * yy)) / w[3]                 
    zz = np.clip(zz, 0, 2)          # Clip Z values for clarity in plot

    # Plot Decsiion Boundary
    ax.plot_surface(xx, yy, zz, alpha = 0.5, color = "k")
    # Scatter plot of data points
    ax.scatter(sorted_data[:50, 0], sorted_data[:50, 1], sorted_data[:50, 2], marker="o", label="Setosa")
    ax.scatter(sorted_data[50:100, 0], sorted_data[50:100, 1], sorted_data[50:100, 2], marker="^", label="Versicolor")
    ax.set_zbound(0, 2)
    ax.legend()
    plt.show()

def wheat_data(epochs, eta):
    # Get Wheat-seeds data
    sorted_data = np.genfromtxt("wheat-seeds.csv", delimiter=",")[:140]
    train = np.vstack((sorted_data[:50], sorted_data[70:120]))
    test = np.vstack((sorted_data[50:70], sorted_data[120:]))
    rng = np.random.default_rng()
    train, test = rng.permutation(train), rng.permutation(test)

    train_x, test_x = train[:, :-1], test[:, :-1]
    train_y, test_y = np.where(train[:, -1] == 1, 1, -1), np.where(test[:, -1] == 1, 1, -1)

    p = Perceptron(eta)
    # Train
    p.fit(train_x, train_y, epochs)

    print(f"{len(test_y) - np.count_nonzero(p.predict(test_x) - test_y)} out of {len(test_y)}")


########### MAIN FUNCTION ############

if __name__ == "__main__":
    # Hyperpararmeters
    epochs = 1000
    eta = 0.5
    
    # True ==> TRAIN PERCEPTRON ON IRIS DATASET AND VISUALISE
    # False ==> TRAIN PERCEPTRON ON WHEAT SEEDS DATAET AND TEST
    iris = True

    if iris:
        iris_data(epochs, eta)
    else:
        wheat_data(epochs, eta)

