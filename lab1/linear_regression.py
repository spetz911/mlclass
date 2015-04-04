# %% Machine Learning Online Class
# %  Exercise 1: Linear regression with multiple variables

import numpy as np
from matplotlib import pyplot


def warmUpExercise():
    A = None
# % ============= YOUR CODE HERE ==============
# % Instructions: Return the 5x5 identity matrix
# %               In octave, we return values by defining which variables
# %               represent the return values (at the top of the file)
# %               and then set them accordingly.

# % ===========================================
    return A


def plotData(X, y):
# % ====================== YOUR CODE HERE ======================
# % Instructions: Plot the training data into a figure using the
# %               "figure" and "plot" commands. Set the axes labels using
# %               the "xlabel" and "ylabel" commands. Assume the
# %               population and revenue data have been passed in
# %               as the x and y arguments of this function.
# %
# % Hint: Попробуй использовать pyplot.scatter,
# % добавлять аргументы отображения на свой вкус,
# % см. описание функции в ipython командой 'pyplot.scatter?'

# правильные названия:
# ylabel('Profit in $10,000s')
# xlabel('Population of City in 10,000s')

# % ============================================================
    return


# %% ======================= Part 2: Plotting =======================

def part2(ex1data1):
    # % Note: You have to complete the code in plotData
    plotData(ex1data1[:, 0], ex1data1[:, 1])
    pyplot.show()


# %% =================== Part 3: Gradient descent ===================

def computeCost(X, y, theta):
    """
    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y
    """
    # % Initialize some useful values
    m = y.shape[0] # % number of training examples
    # % You need to return variable J correctly
    J = None
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta
# %               You should set J to the cost.

# % =========================================================================
    return J



def gradientDescent(X, ymat, theta0, alpha, iterations):
    # % create a copy of theta for simultaneous update.
    theta = theta0
    m = X.shape[0]
    # % number of features.
    p = X.shape[1]

# % Instructions: Perform a single gradient step on the parameter vector theta.
# %
# % Hint: While debugging, it can be useful to print out the values
# %       of the cost function (computeCost) and gradient here.
# %

# % ====================== YOUR CODE HERE ======================
    # % simultaneous update theta using theta_prev.
    for i in range(iterations):
        for j in range(p):
            # % calculate dJ/d(theta_j)
            deriv = None # TODO
            theta[j] = theta[j] - alpha * deriv

# % ============================================================
    return theta



def part3(ex1data1):
    X, y = add_column_of_ones_to_X(ex1data1)
    # здесь используется расширенный набор признаков, поэтому n на 1 больше
    m,n = ex1data1.shape

    theta = np.zeros((n, 1)) # initialize fitting parameters

    # Some gradient descent settings
    num_iters = 1500
    alpha = 0.01

    print("computeCost =", computeCost(X, y, theta))

    theta_res = gradientDescent(X, y, theta, alpha, num_iters)
    print("theta_res =", theta_res[:,0])
    cost_res = computeCostMulti(X, y, theta_res)
    print("cost_res =", cost_res)

    # % Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]).dot(theta_res)
    print('For population = 35,000, we predict a profit of', predict1*10000)
    predict2 = np.array([1, 7]).dot(theta_res)
    print('For population = 70,000, we predict a profit of', predict2*10000)
    
    # Plot outputs
    plotData(X[:,1], y)
    # Plot boundary
    pyplot.plot(X[:,1], (X * theta_res), color='blue')
    pyplot.show()


# %% ================ Part 4: Feature Normalization ================

# %   FEATURENORMALIZE(X) returns a normalized version of X where
# %   the mean value of each feature is 0 and the standard deviation
# %   is 1. This is often a good preprocessing step to do when
# %   working with learning algorithms.
def featureNormalize(X):
    p = X.shape[1]
# % Instructions: First, for each feature dimension, compute the mean
# %               of the feature and subtract it from the dataset,
# %               storing the mean value in mu. Next, compute the
# %               standard deviation of each feature and divide
# %               each feature by it's standard deviation, storing
# %               the standard deviation in sigma.
# %
# %               Note that X is a matrix where each column is a
# %               feature and each row is an example. You need
# %               to perform the normalization separately for
# %               each feature.
# %
# % Hint: You might find the 'mean' and 'std' functions useful.
# % PS в эти функции передается доп аргумент axis=0, например np.mean(matr, axis=0)

# % ====================== YOUR CODE HERE ======================
    X_norm = None
    mu = None
    sigma = None

    X_norm = (X - mu) / sigma

    print("res =", X_norm.shape)
    print("mu =", mu)
    print(X_norm)

# % ============================================================
    return X_norm, mu, sigma





def computeCostMulti(X, y, theta):
# %COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
# %   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
# %   parameter for linear regression to fit the data points in X and y

# % Initialize some useful values
# m = length(y); % number of training examples

# % You need to return the following variables correctly
# J = 0;

# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost of a particular choice of theta
# %               You should set J to the cost.

    J = None # посчитать реальное значение
    return J[0,0]



def gradientDescentMulti(X, y, theta0, alpha, iterations):
    theta = theta0
    m,p = X.shape
    J_history = []
# % Instructions: Perform a single gradient step on the parameter vector
# %               theta.
# %
# % Hint: While debugging, it can be useful to print out the values
# %       of the cost function (computeCostMulti) and gradient here.
# %

# % ====================== YOUR CODE HERE ======================
    for i in range(iterations):
        for j in range(p):
            deriv = None # посчитать производную
            theta[j] = theta[j] - alpha * deriv
# % ============================================================
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history


def part4(ex1data2):
    X,y = add_column_of_ones_to_X(ex1data2)
    m,n = X.shape
    print(X[0])
    print(y)
    print("X.shape =", X.shape)
    # тут не надо нормальизовывать первый столбец
    X_norm0, mu, sigma = featureNormalize(X[:, 1:])
    # возвращаем первый столбец на место
    X_norm = np.concatenate((X[:, 0:1], X_norm0), 1)
    print("X_norm =", X_norm[0])

    # Choose some alpha value
    alpha = 0.1
    num_iters = 400

    # Init Theta and Run Gradient Descent 
    theta0 = np.zeros((3, 1))

    theta_res, J_history = gradientDescentMulti(X_norm, y, theta0, alpha, num_iters)

    print(np.array(J_history))

    # % Plot the convergence graph
    pyplot.plot(range(len(J_history)), J_history, color='blue')
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')
    pyplot.show()

    house = np.matrix([1650, 3]).T
    vec = ((house - mu) / sigma)

    price = 1 * theta_res[0] + vec[0,0] * theta_res[1] + vec[1,0] * theta_res[2]

    print("Predicted price of a 1650 sq-ft, 3 br house")
    print("using gradient descent", price)


def add_column_of_ones_to_X(ex1data1):
    m,n = ex1data1.shape
    tmp1 = np.ones((m, 1))
    tmp2 = np.matrix(ex1data1[:, 0:n-1])
    # print("tmp1.shape =", tmp1.shape)
    # print("tmp2.shape =", tmp2.shape)
    X = np.concatenate((tmp1, tmp2), 1)
    y = np.matrix(ex1data1[:, n-1]).T
    return X, y


# как нарисовать скорость сходимости
# plot curve
# theta_res, J_history = gradientDescentMulti(X, ymat, theta, alpha, num_iters)
# print(np.array(J_history))
# pyplot.plot(range(len(J_history)), J_history, color='blue')
# pyplot.xlabel('Number of iterations')
# pyplot.ylabel('Cost J')
# pyplot.show()


## part X для NORMAL equation

# %NORMALEQN Computes the closed-form solution to linear regression 
# %   NORMALEQN(X,y) computes the closed-form solution to linear 
# %   regression using the normal equations.

# theta = zeros(size(X, 2), 1);

# % ====================== YOUR CODE HERE ======================
# % Instructions: Complete the code to compute the closed form solution
# %               to linear regression and put the result in theta.
# %

# % ---------------------- Sample Solution ----------------------

# theta = pinv(X'*X)*X'*y;

# % -------------------------------------------------------------


# end


def main():
    # убрать коменты с части, которая выполняется в данные момент
    # для part1(warmUpExercise) используется test_sanity
    ex1data1 = np.genfromtxt("ex1data1.txt", delimiter=',')
    part2(ex1data1)
    # part3(ex1data1)
    ex1data2 = np.genfromtxt("ex1data2.txt", delimiter=',')
    # part4(ex1data2)


if __name__ == "__main__":
    np.set_printoptions(threshold=10, linewidth=120, precision=6)
    main()



