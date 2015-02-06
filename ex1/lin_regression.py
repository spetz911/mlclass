import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=10)

# http://wiki.scipy.org/NumPy_for_Matlab_Users

def gradientDescent(Xmat, ymat, theta0, alpha, iterations):
    theta = theta0
    m = ymat.size
    for i in range(iterations):
        p = Xmat.shape[1]
        for j in range(p):
            deriv = (Xmat * theta - ymat).T * Xmat[:, j] / m
            theta[j] = theta[j] - alpha * deriv
    return theta


def computeCostMulti(Xmat, ymat, theta):
    m = ymat.size
    dif = Xmat * theta - ymat
    J = (dif.T * dif) / (2*m)
    return J[0,0]


def gradientDescentMulti(Xmat, ymat, theta0, alpha, iterations):
    theta = theta0
    m = ymat.size
    J_history = []
    for i in range(iterations):
        p = Xmat.shape[1]
        for j in range(p):
            deriv = (Xmat * theta - ymat).T * Xmat[:, j] / m
            theta[j] = theta[j] - alpha * deriv
        J_history.append(computeCostMulti(Xmat, ymat, theta))
    return theta, J_history


# %% ======================= Part 2: Plotting =======================

def part2():
    """    
        ylabel('Profit in $10,000s');
        xlabel('Population of City in 10,000s');
    """
    
    data_tmp = open("ex1data1.txt")
    X_tmp = []
    y_tmp = []
    for row in data_tmp:
        [x0, y0] = map(float, row.strip().split(","))
        X_tmp.append(x0)
        y_tmp.append(y0)
    X_test = np.array(X_tmp, dtype=np.float)
    y_test = np.array(y_tmp, dtype=np.float)

    m = y_test.size

    # X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

    tmp1 = np.ones((m, 1))
    tmp2 = np.mat(X_test).T

    print(tmp1)
    print(tmp2)

    Xmat = np.concatenate((tmp1, tmp2), 1)
    ymat = np.mat(y_test).T

    print(Xmat)

    theta = np.zeros((2, 1)) # initialize fitting parameters

    print(theta)


    # Some gradient descent settings
    num_iters = 1500
    alpha = 0.01

    print(computeCostMulti(Xmat, ymat, theta))


    theta_res = gradientDescent(Xmat, ymat, theta, alpha, num_iters)

    # plot curve
    # theta_res, J_history = gradientDescentMulti(Xmat, ymat, theta, alpha, num_iters)
    # print(np.array(J_history))
    # plt.plot(range(len(J_history)), J_history, color='blue')
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Cost J')
    # plt.show()



    cost_tmp = computeCostMulti(Xmat, ymat, theta_res)

    print("theta_res =", theta_res[:,0])

    print(cost_tmp)

    tmp = (Xmat * theta_res)

    print(tmp)
    
    # Plot outputs
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(Xmat[:,1], (Xmat * theta_res), color='blue')

    plt.show()




def part_multi():
    data_tmp = open("ex1data2.txt")
    mat_tmp = []
    for row in data_tmp:
        nums = list(map(float, row.strip().split(",")))
        mat_tmp.append(nums)
    input_mat = np.matrix(mat_tmp, dtype=np.float)
    X = input_mat[:, 0:2]
    y = input_mat[:, 2]

    m = y.size

    print(X[:, 0])
    xxx = np.ones((m, 1))
    print(xxx)


# %   FEATURENORMALIZE(X) returns a normalized version of X where
# %   the mean value of each feature is 0 and the standard deviation
# %   is 1. This is often a good preprocessing step to do when
# %   working with learning algorithms.

    def featureNormalize(Xmat, ymat):
        p = Xmat.shape[1]
    
        X_norm = Xmat
        mu = np.mean(Xmat, axis=0)
        sigma = np.std(Xmat, axis=0)

        res = (Xmat - mu) / sigma

        print("res =", res.shape)
        print(res)

        # for i in range(p):
        #     X_norm[:,i] = (Xmat[:,i] - mu[i]) / sigma[i]

        return res, mu, sigma


    X_norm, mu, sigma = featureNormalize(X, y)

    Xmat = np.concatenate((xxx, X_norm[:, 0], X_norm[:, 1]), 1)
    ymat = y

    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent 
    theta0 = np.zeros((3, 1))

    theta_res, J_history = gradientDescentMulti(Xmat, ymat, theta0, alpha, num_iters)

    # % Plot the convergence graph
    # figure;
    # plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);

    print(np.array(J_history))

    plt.plot(range(len(J_history)), J_history, color='blue')

    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

    # plt.show()

    house = np.matrix([[1650], [3]])
    vec = ((house - mu) / sigma)

    price = 1 * theta_res[0] + vec[0,0] * theta_res[1] + vec[1,0] * theta_res[2]

    print("Predicted price of a 1650 sq-ft, 3 br house")
    print("using gradient descent", price)



def main():
    part_multi()


main()
