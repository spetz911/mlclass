import numpy as np
import scipy.special
import scipy.optimize
import scipy.io
from matplotlib import pyplot


def plot_data(data):
    positives = data[data[:,2] == 1]
    negatives = data[data[:,2] == 0]
    pyplot.xlabel("Exam 1 score")
    pyplot.ylabel("Exam 2 score")
    pyplot.xlim([25, 115])
    pyplot.ylim([25, 115])
    pyplot.scatter( negatives[:, 0], negatives[:, 1],
        c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
    pyplot.scatter( positives[:, 0], positives[:, 1],
        c='b', marker='+', s=40, linewidths=2, label="Admitted" )
    pyplot.legend()



def findMinTheta(theta0, X, y, lamda):
    result = scipy.optimize.minimize(costFunction, x0=theta0, args=(X, y, lamda),
        method="BFGS", options={'maxiter': 400, 'disp': True})
    return result.x, result.fun


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



def computeCost( theta, X, y, lamda ):
    m           = X.shape[0]
    term1       = X.dot( theta ) - y 
    left_term   = term1.T.dot( term1 ) / (2 * m)
    right_term  = theta[1:].T.dot( theta[1:] ) * (lamda / (2*m))
    J = (left_term + right_term).flatten()[0]
    return J


def computeGradient(theta, X, y, lamda):
    m           = X.shape[0]
    grad        = X.dot( theta ) - y 
    grad        = X.T.dot( grad) / m
    grad[1:]    = grad[1:] + theta[1:] * lamda / m

    return grad.flatten()



def task0():
    # print(ex5data.shape)
    X = ex5data['X']
    m,n = X.shape
    y = ex5data['y']
    Xval = ex5data['Xval']
    yval = ex5data['yval']
    Xtest = ex5data['Xtest']
    ytest = ex5data['ytest']
    print(X.shape, y.shape)
    print(Xval.shape, yval.shape)
    
    # plotData(X,y)

    theta0 = np.ones((n+1, 1))
    ones_col = np.ones((m, 1))
    lamda = 1.0
    Xtrain = np.c_[ones_col, X  ]
    initCost = computeCost(theta0, Xtrain, y, lamda)
    print(initCost)

    initGradient = computeGradient(theta0, Xtrain, y, lamda)
    print(initGradient)


# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset. 
# %  The following code will load the dataset into your environment and plot
# %  the data.
# %

def plotData(X, y):
    pyplot.scatter( X, y, marker='x', c='r', s=30, linewidth=2 )
    pyplot.xlim([X.min(), X.max()])
    pyplot.ylim([y.min(), y.max()])
    pyplot.xlabel('Change in water level(x)')
    pyplot.ylabel('Water flowing out of the dam(y)')


def part1(ex5data):
    X = ex5data['X']
    m,n = X.shape
    y = ex5data['y']
    plotData(X, y)
    pyplot.show()


# %% =========== Part 2: Regularized Linear Regression Cost =============
# %  You should now implement the cost function for regularized linear 
# %  regression. 
# %

def computeCostFunction(theta0, X, y, lamda):
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost and gradient of regularized linear 
# %               regression for a particular choice of theta.
# %
# %               You should set J to the cost and grad to the gradient.
# %

    theta = np.matrix(np.ravel(theta0)).T
    m = X.shape[0]
    # ...
    J = None
# % =========================================================================
    return J


# %% =========== Part 3: Regularized Linear Regression Gradient =============
# %  You should now implement the gradient for regularized linear 
# %  regression.
# %

def computeGradient(theta0, X, y, lamda):
# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the cost and gradient of regularized linear 
# %               regression for a particular choice of theta.
# %
# %               You should set J to the cost and grad to the gradient.
# %

    theta = np.matrix(np.ravel(theta0)).T
    m = X.shape[0]
    # ...
    grad = None
# % =========================================================================
    return grad.ravel()



# %% =========== Part 4: Train Linear Regression =============
# %  Once you have implemented the cost and gradient correctly, the
# %  trainLinearReg function will use your cost function to train 
# %  regularized linear regression.
# % 
# %  Write Up Note: The data is non-linear, so this will not give a great 
# %                 fit.
# %

def trainLinearReg(theta0, X, y, lamda):
    result = scipy.optimize.minimize(computeCostFunction, x0=theta0, args=(X, y, lamda),
        method="BFGS", options={'maxiter': 400, 'disp': False})
    return result.x, result.fun


def part4(ext_X, y):
    lamda = 0
    theta0 = np.zeros((ext_X.shape[1], 1))
    theta_min, cost = trainLinearReg(theta0, ext_X, y, lamda)
    print("theta =", theta_min)

    plotData(ext_X[:, 1:2], y)
    pyplot.plot(ext_X[:, 1], ext_X * np.matrix(theta_min).T)
    pyplot.show()


# %% =========== Part 5: Learning Curve for Linear Regression =============
# %  Next, you should implement the learningCurve function. 
# %
# %  Write Up Note: Since the model is underfitting the data, we expect to
# %                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
# %

def part5(ext_Xval, yval, ext_X, y):
    lamda = 0.0
    learningCurve(ext_X, y, ext_Xval, yval, lamda)


def learningCurve(X_train, y_train, X_val, y_val, lamda):
# %LEARNINGCURVE Generates the train and cross validation set errors needed 
# %to plot a learning curve
# %   [error_train, error_val] = ...
# %       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
# %       cross validation set errors for a learning curve. In particular, 
# %       it returns two vectors of the same length - error_train and 
# %       error_val. Then, error_train(i) contains the training error for
# %       i examples (and similarly for error_val(i)).
# %
# %   In this function, you will compute the train and test errors for
# %   dataset sizes from 1 up to m. In practice, when working with larger
# %   datasets, you might want to do this in larger intervals.
    m,n = X_train.shape
    res_cv = []
    res_train = []
    theta0 = np.zeros((X_train.shape[1], 1))
# =========================================================================
# % Instructions: Fill in this function to return training errors in 
# %               error_train and the cross validation errors in error_val. 
# %               i.e., error_train(i) and 
# %               error_val(i) should give you the errors
# %               obtained after training on i examples.
# %
# % Note: You should evaluate the training error on the first i training
# %       examples (i.e., X(1:i, :) and y(1:i)).
# %
# %       For the cross-validation error, you should instead evaluate on
# %       the _entire_ cross validation set (Xval and yval).
# %
# % Note: If you are using your cost function (linearRegCostFunction)
# %       to compute the training and cross validation error, you should 
# %       call the function with the lambda argument set to 0. 
# %       Do note that you will still need to use lambda when running
# %       the training to obtain the theta parameters.
# %
# % Hint: You can loop over the examples with the following:
# %
# %       for i = 1:m
# %           % Compute train/cross validation errors using training examples 
# %           % X(1:i, :) and y(1:i), storing the result in 
# %           % error_train(i) and error_val(i)
# %           ....
# %           
# %       end
# %

# % ====================== YOUR CODE HERE ======================
# следует заменить None на корректные выражения
    start_i = 3
    for i in range(start_i, m):
        # часть тренировочной выборки для построения графика
        curr_X = X_train[:i]
        curr_y = y_train[:i]
        # обучаем theta_min на выборке
        theta_min, _cost = None # use trainLinearRegtrain(...)
        # вычисляем ошибку обучения по всей обучающей выборке
        train_error = None # use computeCostFunction(...)
        res_train.append(train_error)
        # вычисляем ошибку на выборке для кросс-валидации
        cv_error = None # use computeCostFunction(...)
        res_cv.append(cv_error)
# % =========================================================================

    pyplot.title('Learning curve for linear regression')
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.plot(np.arange(len(res_cv))+start_i, res_cv)
    # pyplot.legend()
    pyplot.plot(np.arange(len(res_train))+start_i, res_train)
    pyplot.legend(['Train', 'Cross Validation'])

    # pyplot.ylim([-1, 60])

    pyplot.show()


# %% =========== Part 6: Feature Mapping for Polynomial Regression =============
# %  One solution to this is to use polynomial regression. You should now
# %  complete polyFeatures to map each example into its powers
# %


def polyFeatures(x1, degree):
# %POLYFEATURES Maps X (1D vector) into the p-th power
# %   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
# %   maps each example into its polynomial features where
# %   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
# %
    m = x1.size
    # print(x1)
    # print(x1.ravel())
    X = np.zeros((m, degree))
# % ====================== YOUR CODE HERE ======================
# % Instructions: Given a vector X, return a matrix X_poly where the p-th 
# %               column of X contains the values of X to the p-th power.
# %
# % 
# % =========================================================================
    # print("DEBUG")
    # print(X.shape)
    # print(X)
    return X


def featureNormalize(X, mu, sigma):
    Xnorm0 = (X - mu) / sigma
    tmp = np.ones((Xnorm0.shape[0], 1))
    Xnorm = np.concatenate((tmp, Xnorm0), 1)
    return Xnorm


def part6(X0, Xval1, Xtest1, degree):
    Xpoly = polyFeatures(X0, degree)
    mu = np.mean(Xpoly, axis=0)
    sigma = np.std(Xpoly, axis=0)
    X_train = featureNormalize(Xpoly, mu, sigma)
    X_val_poly = polyFeatures(Xval1, degree)
    X_val = featureNormalize(X_val_poly, mu, sigma)
    X_test_poly = polyFeatures(Xtest1, degree)
    X_test = featureNormalize(X_test_poly, mu, sigma)
    print("X_train[0] =", X_train[0])
    return X_train, X_val, X_test, mu, sigma


# %% =========== Part 7: Learning Curve for Polynomial Regression =============
# %  Now, you will get to experiment with polynomial regression with multiple
# %  values of lambda. The code below runs polynomial regression with 
# %  lambda = 0. You should try running the code with different values of
# %  lambda to see how the fit and learning curve change.
# %

def part7(X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma, X0, degree):
    # TODO менять lamda параметр по вкусу
    lamda = 0.0
    print(X_train.shape)
    theta0 = np.zeros((X_train.shape[1], 1))
    theta, _cost = trainLinearReg(theta0, X_train, y_train, lamda)
    print("theta (!) =", theta)
    x_axis0 = np.arange(np.min(X0)-5, np.max(X0)+5, 0.1)
    x_axis = np.matrix(x_axis0).T
    x_poly = polyFeatures(x_axis, degree)
    x_norm = featureNormalize(x_poly, mu, sigma)
    y_axis = x_norm * np.matrix(theta).T
    # закоментировать график чтоб не мешал строить кривую обучения
    pyplot.scatter(X0, y_train)
    pyplot.plot(x_axis, y_axis)
    pyplot.show()
    # plot Learn Curve
    learningCurve(X_train, y_train, X_val, y_val, lamda)


# %% =========== Part 8: Validation for Selecting Lambda =============
# %  You will now implement validationCurve to test various values of 
# %  lambda on a validation set. You will then use this to select the
# %  "best" lambda value.
# %

# %VALIDATIONCURVE Generate the train and validation errors needed to
# %plot a validation curve that we can use to select lambda
# %   [lambda_vec, error_train, error_val] = ...
# %       VALIDATIONCURVE(X, y, Xval, yval) returns the train
# %       and validation errors (in error_train, error_val)
# %       for different values of lambda. You are given the training set (X,
# %       y) and validation set (Xval, yval).
# %

def validationCurve(lamda_vec, X_train, y_train, X_val, y_val):
    # % You need to return these variables correctly.
    err_cv = []
    err_train = []
    theta0 = np.zeros((X_train.shape[1], 1))
# % Instructions: Fill in this function to return training errors in 
# %               error_train and the validation errors in error_val. The 
# %               vector lambda_vec contains the different lambda parameters 
# %               to use for each calculation of the errors, i.e, 
# %               error_train(i), and error_val(i) should give 
# %               you the errors obtained after training with 
# %               lambda = lambda_vec(i)
# %
# % Note: You can loop over lambda_vec with the following:
# %
# %       for i = 1:length(lambda_vec)
# %           lambda = lambda_vec(i);
# %           % Compute train / val errors when training linear 
# %           % regression with regularization parameter lambda
# %           % You should store the result in error_train(i)
# %           % and error_val(i)
# %           ....
# %           
# %       end
# %
# %
# % ====================== YOUR CODE HERE ======================
# исправить строки с None
    for i in range(len(lamda_vec)):
        theta_min, _cost = None # trainLinearReg(...)
        train_error = None # computeCostFunction(...)
        err_train.append(train_error)
        cv_error = None # computeCostFunction(...)
        err_cv.append(cv_error)
# % =========================================================================
    return err_cv, err_train


def part8(X_train, y_train, X_val, y_val):
    # % Selected values of lambda (you should not change this)
    lamda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 4, 5, 10]
    err_cv, err_train = validationCurve(lamda_vec, X_train, y_train, X_val, y_val)
    pyplot.title('lambda\t\tTrain Error\tValidation Error')
    pyplot.xlabel('lambda')
    pyplot.ylabel('Error')
    pyplot.plot(lamda_vec, err_cv)
    # pyplot.legend()
    pyplot.plot(lamda_vec, err_train)
    pyplot.legend(['Train', 'Cross Validation'])
    pyplot.show()


def main():
    np.set_printoptions(threshold=10, linewidth=120, precision=6)
    ex5data = scipy.io.loadmat("ex5data1.mat")
    # part1(ex5data)

    X1 = ex5data['X']
    Xval1 = ex5data['Xval']
    Xtest1 = ex5data['Xtest']
    y_train = ex5data['y']
    y_val = ex5data['yval']
    y_test = ex5data['ytest']

    tmp = np.ones((X1.shape[0], 1))
    ext_X = np.concatenate((tmp, X1), 1)
    tmp = np.ones((Xval1.shape[0], 1))
    ext_Xval = np.concatenate((tmp, Xval1), 1)
    m,n = ext_X.shape

    # для части 2 и 3 используется test_sanity

    # part4(ext_X, y_train)
    # part5(ext_Xval, y_val, ext_X, y_train)

    degree = 8
    X_train, X_val, X_test, mu, sigma = part6(X1, Xval1, Xtest1, degree)
    # part7(X_train, y_train, X_val, y_val, X_test, y_test, mu, sigma, X1, degree)
    part8(X_train, y_train, X_val, y_val)

    lamda_test = 3.0
    theta0 = np.zeros((X_train.shape[1], 1))
    theta_min, _cost = trainLinearReg(theta0, X_train, y_train, lamda_test)
    test_error = computeCostFunction(theta_min, X_test, y_test, 0.0)
    print("test_error =", test_error)
    cv_error = computeCostFunction(theta_min, X_val, y_val, 0.0)
    print("cv_error =", cv_error)
    # TODO написать в отчете, почему полученная ошибка меньше чем в задании


if __name__ == "__main__":
    main()










