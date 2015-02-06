
import numpy as np
import scipy.special
import scipy.optimize
from matplotlib import pyplot

np.set_printoptions(threshold=10)

def load_csv(filename):
    input_data = []
    with open(filename) as rr:
        for line in rr:
            row_nums = list(map(float, line.strip().split(",")))
            input_data.append(row_nums)
    return np.matrix(input_data)


def plot_data(data):
    positives  = data[data[:,2] == 1]
    negatives  = data[data[:,2] == 0]

    pyplot.xlabel("Exam 1 score")
    pyplot.ylabel("Exam 2 score")
    pyplot.xlim([25, 115])
    pyplot.ylim([25, 115])

    pyplot.scatter( negatives[:, 0], negatives[:, 1],
        c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
    pyplot.scatter( positives[:, 0], positives[:, 1],
        c='b', marker='+', s=40, linewidths=2, label="Admitted" )

    pyplot.legend()





def costFunction(theta0, Xmat, yvec):
    m = yvec.size

    theta = np.matrix(theta0).T

    # calculate cost function
    # hx = 1.0 / (1 + np.exp(-1 * Xmat * theta))

    # print("-----")
    # print("Xmat =", Xmat.shape)
    # print("yvec =", yvec.shape)
    # print("theta =", theta.shape)

    tmp0 = Xmat * theta
    hx = scipy.special.expit(tmp0)

    # print("hx =", hx)

    tmp1 = (0 - yvec.T) * np.log(hx)
    tmp2 = (1 - yvec.T) * np.log(1-hx)

    J_cost = (tmp1 - tmp2) / m
    # gradient = (Xmat.T * (hx - yvec)) / m

    return J_cost


def findMinTheta(theta0, X, y):
    result = scipy.optimize.fmin(costFunction, x0=theta0, args=(X, y), maxiter=400, full_output=True)
    return result[0], result[1]


def plotDecisionBoundary(ex2data, X, y, theta):
    # theta1 = theta.flatten()
    plot_data(ex2data)
    plot_x = np.arange(X.min(), X.max())
    # line equation with a + b*x + cy = 0
    plot_y = -1.0 * (theta[1] * plot_x + theta[0]) / theta[2]
    pyplot.plot( plot_x, plot_y )


def predict(X, theta0):
    theta = np.matrix(theta0.flatten()).T
    hx = scipy.special.expit(X * theta)
    m = X.shape[0]
    prob = np.zeros((m,1))
    for i in range(m):
        prob[i] = 1 if hx[i] >= 0.5 else 0
    return prob


def task1():
    ex2data = np.genfromtxt("ex2data1.txt", delimiter=',')
    print("mat1 =", ex2data)

    # plot_data(ex2data)
    # pyplot.show()

    m, n = ex2data.shape
    ones_col = np.ones((m, 1))
    Xmat = np.matrix(np.c_[ ones_col, ex2data[:,0:2] ])
    ymat = np.matrix(ex2data[:, 2]).T
    print("Xmat =", Xmat.shape)
    print("ymat =", ymat.shape)

    # % Initialize fitting parameters
    initial_theta = np.zeros((n, 1))

    # % Compute and display initial cost and gradient
    
    J_cost = costFunction(initial_theta.T, Xmat, ymat)
    
    print("J_cost =", J_cost)
    
    #     Cost at initial theta (zeros): 0.693147
    #     Gradient at initial theta (zeros): 
    #     -0.100000 
    #     -12.009217 
    #     -11.262842 

    theta, cost = findMinTheta(initial_theta.T, Xmat, ymat)
    
    # % Print theta to screen
    print("Cost at theta found by fminunc:", cost)
    print("theta:", theta)

    # % Plot Boundary
    # plotDecisionBoundary(ex2data, Xmat, ymat, theta)
    # pyplot.show()

    prob = scipy.special.expit( theta * np.matrix([1, 45, 85]).T)
    print("For a student with scores 45 and 85,",
          "we predict an admission probability of",
          100 * prob[0,0])
    
    p = predict(Xmat, theta)
    accuracy = np.mean(np.double(p == ymat))
    print('Train Accuracy:', accuracy)









def main():
    task1()






if __name__ == "__main__":
    main()
































