
import numpy as np
import scipy.special
import scipy.optimize
from matplotlib import pyplot


def plot_data(data):
    positives = data[data[:,2] == 1]
    negatives = data[data[:,2] == 0]
    pyplot.xlabel('Microchip Test 1')
    pyplot.ylabel('Microchip Test 2')
    pyplot.scatter( negatives[:, 0], negatives[:, 1],
        c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
    pyplot.scatter( positives[:, 0], positives[:, 1],
        c='b', marker='+', s=40, linewidths=2, label="Admitted" )
    pyplot.legend()


def plotDecisionBoundary(ex2data, X, y, theta):
    plot_data(ex2data)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            mapped = mapFeatures(u[i], v[j])
            # print("tmp0.shape =", tmp0.shape)
            # tmp01 = np.matrix(theta.flatten()).T
            tmp1 = mapped.dot(theta)
            # print("tmp1.shape =", tmp1.shape)
            z[i, j] = tmp1
    z = z.T

    pyplot.contour(u, v, z, [0, 0], linewidth=1)


def plottXXX():
        u = linspace( -1, 1.5, 50 )
        v = linspace( -1, 1.5, 50 )
        z = zeros( (len(u), len(v)) )

        for i in range(0, len(u)): 
            for j in range(0, len(v)):
                mapped = mapFeature( array([u[i]]), array([v[j]]) )
                z[i,j] = mapped.dot( theta )
        z = z.transpose()

        u, v = meshgrid( u, v ) 
        pyplot.contour( u, v, z, [0.0, 0.0], label='Decision Boundary' )        

        pyplot.show()

def paramsFunction(theta0, Xmat):
    theta = np.matrix(theta0.flatten()).T
    theta1 = theta.copy()
    theta1[0] = 0.0
    tmp0 = Xmat * theta
    hx = scipy.special.expit(tmp0)
    return hx, theta1

    # print("-----")
    # print("Xmat =", Xmat.shape)
    # print("yvec =", yvec.shape)
    # print("theta =", theta.shape)
    # print("hx =", hx)

def costFunction(theta0, X, y, lam):
    m = X.shape[0]
    hx, theta1 = paramsFunction(theta0, X)
    p = lam * (theta1.T * theta1) / (2*m)
    tmp1 = (0 - y.T) * np.log(hx)
    tmp2 = (1 - y.T) * np.log(1 - hx)
    J_cost = p + (tmp1 - tmp2) / m
    return J_cost

def gradientFunction(theta0, X, y, lam):
    hx, theta1 = paramsFunction(theta0, X)
    p1 = lam * theta1
    gradient = (Xmat.T * (hx - y) + p1) / m
    return gradient



def findMinTheta(theta0, X, y, lam):
    result = scipy.optimize.minimize(costFunction, x0=theta0, args=(X, y, lam),
        method="BFGS", options={'maxiter': 400, 'disp': True})
    return result.x, result.fun


def mapFeatures(x1, x2):
    m = x1.size
    degree = 6
    row_size = (degree + 2) * (degree + 1) // 2
    X = np.zeros((m, row_size))
    col_num = 0
    for i in range(0, degree+1):
        for j in range(0, i+1):
            # print(i-j, j)
            X[:, col_num] = np.power(x1, i-j) * np.power(x2, j)
            col_num += 1
    # print("DEBUG")
    # print(X.shape)
    # print(X)
    return X




def predict(X, theta0):
    theta = np.matrix(theta0.flatten()).T
    hx = scipy.special.expit(X * theta)
    m = X.shape[0]
    prob = np.zeros((m,1))
    for i in range(m):
        prob[i] = 1 if hx[i] >= 0.5 else 0
    return prob





# degree = 6;
# out = ones(size(X1(:,1)));
# for i = 1:degree
#     for j = 0:i
#         out(:, end+1) = (X1.^(i-j)).*(X2.^j);
#     end
# end


def task1():
    ex2data = np.genfromtxt("ex2data2.txt", delimiter=',')
    print("mat1 =", ex2data)

    # plot_data(ex2data)
    # pyplot.show()

    # ones_col = np.ones((m, 1))
    # Xmat = np.matrix(np.c_[ ones_col, ex2data[:, 0:n-1] ])
    x1 = ex2data[:,0]
    x2 = ex2data[:,1]
    X = np.matrix(mapFeatures(x1, x2))
    print("mappedFeatures = ")
    y = np.matrix(ex2data[:, -1].flatten()).T
    m, n = X.shape
    print("Xmat =", X.shape)
    print("ymat =", y.shape)

    # % Initialize fitting parameters
    initial_theta = np.zeros((n, 1))

    # % Compute and display initial cost and gradient
    lam = 1.0
    J_cost = costFunction(initial_theta.T, X, y, lam)
    print("J_cost =", J_cost)
    
    #     Cost at initial theta (zeros): 0.693147
    #     Gradient at initial theta (zeros): 
    #     -0.100000 
    #     -12.009217 
    #     -11.262842 

    theta, cost = findMinTheta(initial_theta.T, X, y, lam)
    
    # % Print theta to screen
    print("Cost at theta found by fminunc:", cost)
    print("theta:", theta)

    # % Plot Boundary
    plotDecisionBoundary(ex2data, X, y, theta)
    pyplot.show()
    
    p = predict(X, theta)
    accuracy = np.mean(np.double(p == y))
    print('Train Accuracy:', accuracy)









def main():
    np.set_printoptions(threshold=10, linewidth=120, precision=6)
    task1()






if __name__ == "__main__":
    main()
































