"""
COMP575 - Coursework (MLP)
--------------------------
By Thomas Rex Greenway 201198319

Implementation of MLP with multiple training methodologies (Backpropation,
Genetic Algorithm, and Particle Swarm Optimisation).

The network is trained on 1 of 2 randomly generated datasets:
    
    - SIMPLE: 2D Vector Sum Approximation
        Input data is an array of randomly chosen real numbers in [0, 1), 
        [x_1, y_1, x_2, x_2], representing 2 vectors, (x_1, y_1) and
        (x_2, y_2). Target data is the corresponding vector sum 
        (x_1 + x_2, y_1 + y_2).
    
    - COMPLEX: Finding the Centere of a Sphere
        Input data is a flattened array of 30 randomly generated points on
        the surface of a sphere with radius 1 and given, target value,
        centre. The target intger centres are randomly generatd with 
        x, y, z = 0 or 1.
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt


############ MLP CLASS ################

class MLP():
    """
    Multi-Layer Perceptron Neural Network with given structure and
    activation methods.

    Parameters
    ----------
    num_inputs : int
        Number of values the MLP takes as an input.
    hidden_layers : list(int)
        List of integers corresponding to hidden layers within the network.
    num_outputs : int
        Number of values the MLP produces as an output.
    weights : NumPy Array
        Flattened weight array for use in GA and PS Optimisation.
    activation_method : "sigmoid", "tanh", or "relu"
        String describing the activation function to be used in the network 
        neurons
    """
    def __init__(self, num_inputs, hidden_layers, num_outputs, weights = [], activation_method = "sigmoid"):
        """
        Initialises a MLP Neural Network with given structure.
        """
        # Layer Structure
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # Layer Activation Method
        self.activation_method = activation_method

        # NN Structure
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # Create list of weight matrices between each layer
        self.weights = []
        for i in range(len(layers) - 1):
            # Weight initialisation given flattened weight array.
            if len(weights) != 0:
                w = np.zeros((layers[i], layers[i + 1]))
                for j in range(layers[i]):
                    w[j, :] = weights[(j * layers[i + 1]) : (j * layers[i + 1]) + layers[i + 1]]
            # Else random inital weights
            else:
                w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        # Derivative list for each weight matrix.
        self.derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            self.derivatives.append(d)

        # Activations list for each layer.
        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

    def forward_propagate(self, inputs):
        """
        Forward pass of given inputs through the network.

        Returns
        -------
        activations : NumPy Array (1 x num_ouput)
            Final Activation values for each node in output layer.
        """
        # Input Layer
        activations = inputs
        self.activations[0] = activations

        # For each layer in network (sans Input Layer)
        for i, w in enumerate(self.weights):
            # Net Input into layer
            net_inputs = np.dot(activations, w)
            # Apply activation function
            activations = self.activate(net_inputs)
            # Stor layers Activation array
            self.activations[i + 1] = activations
        return activations

    def back_propagate(self, error):
        """
        Backward pass though the newtork to update derivatives matrices w.r.t error.
        """
        # Iterate back through each layer in the network.
        for i in reversed(range(len(self.derivatives))):
            # Activation for previous layer
            activations = self.activations[i+1]
            # Activation for current layer
            current_activations = self.activations[i]

            # Output Layer : delta = total_Error * a_prime(net_input)
            # Input Layer : delta = sum(prev_deltas * weights) * a_prime(net_input)
            delta = error * self.activate_derivative(activations)

            # Reshape delta and current activations for matrix multiplications
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # dE/dW = delta * activations
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # Change error for hidden layers (see above comment)
            error = np.dot(delta, self.weights[i].T)

    def fit(self, inputs, targets, epochs, learning_rate):
        """
        Train the neural network using forward- and back-propagation on given
        training inputs and targets.
        """
        # Store epoch errors for plotting
        self.errors = np.zeros(epochs)

        # Iterate through epochs
        for i in range(epochs):
            # Error for current epoch
            sum_errors = 0

            # Iterate through training inputs
            for j, input in enumerate(inputs):
                # Get Output (forward pass), Target,  and thus error:
                target = targets[j]
                output = self.forward_propagate(input)
                error = target - output

                # Back-prop. updates derivatives + Update Weights
                self.back_propagate(error)
                self._gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)

            self.errors[i] = sum_errors

            # Print iteration tracking to console
            if i % min(100, int(epochs / 10)) == 0:
                print(f"Iteration: {i}\nCurrent Error {sum_errors/len(inputs)}")
        print(f"\nFinal Error: {sum_errors/len(inputs)}")

    def _gradient_descent(self, learningRate=1):
        """
        Updates internal network weights using derivatives generated in backward
        pass. [Gradient Descent]
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    ## ACTIVATION FUNCTION SELECTOR ##
    def activate(self, x):
        if self.activation_method == "sigmoid":
            return self._sigmoid(x)
        elif self.activation_method == "tanh":
            return self._tanh(x)
        elif self.activation_method == "relu":
            return self._relu(x)

    def activate_derivative(self, x):
        if self.activation_method == "sigmoid":
            return self._sigmoid_derivative(x)
        elif self.activation_method == "tanh":
            return self._tanh_derivative(x)
        elif self.activation_method == "relu":
            return self._relu_derivative(x)
    
    ## ACTIVATION FUNCTIONS ##
    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        a_prime = np.where(x > 0, 1, 0)
        return a_prime

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    ## TOTAL ERROR ##
    def _mse(self, target, output):
        """
        Get Mean Square Error between two points.
        """
        return np.average((target - output) ** 2)



############## UPDATE FUNCTIONS (BackProp., GA, PSO) ##################

def nn_update(mlp_structure, train_x, train_y):
    """
    Trains a multi-layer perceptron neural network using backpropagation to
    update weights and learn given training data.     
    """
    # MLP Structure
    num_inputs = mlp_structure[0]
    num_hidden = [i for i in mlp_structure[1:-1]]
    num_ouputs = mlp_structure[-1]
    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(num_inputs, num_hidden, num_ouputs, activation_method="tanh")
    # train network
    mlp.fit(train_x, train_y, 1000, 0.1)
    # plot
    plt.plot(mlp.errors / len(train_x))
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.show()
    return mlp

def ga_update(mlp_structure, train_x, train_y):
    """
    Performs Genetic Algorithm Optimisation to train neural network weights to
    learn given training data.     
    """
    # IMPORT GA Library
    from geneal.genetic_algorithms import ContinuousGenAlgSolver
    # MLP Structure
    num_inputs = mlp_structure[0]
    num_hidden = [i for i in mlp_structure[1:-1]]
    num_ouputs = mlp_structure[-1]

    # SUB FUNCTIONS
    def get_n_genes(nn_structure):
        sol_shape = 0
        for i in range(len(mlp_structure) - 1):
            sol_shape += mlp_structure[i] * mlp_structure[i + 1]
        return sol_shape

    def fitness(X):
        # create mlp with weights
        mlp_sol = MLP(num_inputs, num_hidden, num_ouputs, weights = X)
        # foward prop that mlp on training data
        outputs = mlp_sol.forward_propagate(train_x)
        # Returns mean squared error for that mlp solution
        return - np.average((train_y - outputs) ** 2)

    # GA Algorithm
    n_genes = get_n_genes(mlp_structure)
    solver = ContinuousGenAlgSolver(
        n_genes=n_genes, 
        fitness_function=fitness,
        pop_size=50,
        max_gen=1000,
        mutation_rate=0.01,         
        selection_rate=0.1,         
        selection_strategy="roulette_wheel",
        problem_type=float, 
        variables_limits=(-2, 2)   
    )
    solver.solve()
    sol = solver.best_individual_

    # MLP with optimized weights
    mlp = MLP(num_inputs, num_hidden, num_ouputs, weights = sol)
    return mlp

def pso_update(mlp_structure, train_x, train_y):
    """
    Performs Particle Swarm Optimisation to train neural network weights to
    learn given training data.     
    """
    # Import PySwarms
    import pyswarms as ps
    from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

    num_inputs = mlp_structure[0]
    num_hidden = [i for i in mlp_structure[1:-1]]
    num_ouputs = mlp_structure[-1]

    def get_n_genes(nn_structure):
        sol_shape = 0
        for i in range(len(mlp_structure) - 1):
            sol_shape += mlp_structure[i] * mlp_structure[i + 1]
        return sol_shape

    def fitness(X):
        # create mlp with weights
        mlp_sol = MLP(num_inputs, num_hidden, num_ouputs, weights = X)
        # foward prop that mlp on training data
        outputs = mlp_sol.forward_propagate(train_x)
        # Returns mean squared error for that mlp solution
        return np.average((train_y - outputs) ** 2)
    
    def f(X):
        # Function to do mass fitness on all particles
        n_particles = X.shape[0]
        j = [fitness(X[i]) for i in range(n_particles)]
        return np.array(j)

    num_particles = 100
    n_genes = get_n_genes(mlp_structure)
    options = {'c1': 0.9, 'c2': 0.1, 'w':0.9}           # 0.5, 0.3, 0.9
    bounds = (np.array([-10] * n_genes), np.array([10] * n_genes))
    oh_strategy = {"c1":"nonlin_mod", "c2":'lin_variation'}

    # Perform optimization and plot cost func.
    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles,
        dimensions=n_genes,
        options=options,
        bounds=bounds,
        oh_strategy=oh_strategy
        )
    cost, pos = optimizer.optimize(f, iters=1000)
    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    # MLP with optimized weights
    mlp = MLP(num_inputs, num_hidden, num_ouputs, weights = pos)
    return mlp



######### DATA GENERATION #############

def get_rand_points_on_sphere(num_points, center, r):
    """
    Generates given number of random points on the surface of a sphere
    with given center and radius.
    """
    rgen = np.random.default_rng()
    sphere = np.zeros((num_points, 3))
    for i in range(num_points):
        phi = rgen.uniform(high = 2 * np.pi)
        theta = rgen.uniform(high = np.pi)
        x = r * np.cos(phi) * np.sin(theta) + center[0]
        y = r * np.sin(phi) * np.sin(theta) + center[1]
        z = r * np.cos(theta) + center[2]
        sphere[i, 0] = x
        sphere[i, 1] = y
        sphere[i, 2] = z
    return sphere

def generate_training_data(training_size, dimension, center_range, radius):
    """
    Generates 2 arrays: 
        train_y : Given number of randomly generated integer center points 
            within certain range.
        train_x : Corresponding Array of random points on sphere of given 
            radius about each center point.
    """
    train_x = np.empty((0, dimension * 3))
    train_y = np.empty((0, 3))
    for i in range(training_size):
        # Gen. center
        center = np.random.randint(center_range[0], center_range[1], size=3)
        # Store center point
        train_y = np.vstack((train_y, center))
        # Gen Sphere, flatten, and add to training data
        sphere = get_rand_points_on_sphere(dimension, center, radius)
        flat_sphere = sphere.flatten()
        train_x = np.vstack((train_x, flat_sphere))

    return train_x, train_y



######### MAIN FUNCTION ##########

def main(method, mlp_structure, data):
    # TRAIN
    if data == "sphere":
        training_size = 1000
        sphere_points = 30
        center_range = (0, 2)
        radius = 1
        train_x, train_y = generate_training_data(training_size, sphere_points, center_range, radius)

        # # PLOT FIRST SPHERE FOR SHOW
        # import matplotlib.pyplot as plt
        # test = train_x[0]
        # target = train_y[0]
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(test[0:sphere_points*3:3], test[1:sphere_points*3:3], test[2:sphere_points*3:3], marker="o")
        # ax.scatter(target[0], target[1], target[2], marker="^")
        # plt.show()
    
    else:
        # VECTOR SUM TRAINING
        train_x = np.array([[np.random.random()/2 for _ in range(4)] for _ in range(1000)])
        train_y = np.array([[i[0] + i[2], i[1] + i[3]] for i in train_x])

    # METHOD SELECTOR
    if method == "nn":
        return nn_update(mlp_structure, train_x, train_y)
    elif method == "ga":
        return ga_update(mlp_structure, train_x, train_y)
    elif method == "pso":
        return pso_update(mlp_structure, train_x, train_y)
    else:
        print("NO SUCH UPDATE RULE")


if __name__ == "__main__":
    # METHOD + DATA
    method = "pso"
    data = "sphere"

    # MLP Structure (Dependent on data)
    if data == "sphere":
        sphere_points = 30
        center_range = (0, 2)
        radius = 1
        mlp_structure = [sphere_points * 3, 32, 3]
    else:
        mlp_structure = [4, 5, 2]
        
    # TRAINING
    mlp = main(method, mlp_structure, data)

    # TEST
    if data == "sphere":
        y = np.random.randint(center_range[0], center_range[1], size=3)
        X = get_rand_points_on_sphere(sphere_points, y, radius)
    else:
        # VECTOR SUM TEST
        X = np.array([0.3, 0.1, 0.2, 0.1])
        y = np.array([0.5, 0.2])
    
    # SINGLE RANDOM TEST
    output = mlp.forward_propagate(X.flatten())
    print(f"\n\tEXPECTED: {y} --> PREDICTED: {np.round_(output, 5)}")
    