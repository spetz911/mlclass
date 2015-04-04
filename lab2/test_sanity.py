import unittest
import regularized_linear
import numpy as np
import scipy.io


class LinearRegression(unittest.TestCase):

    def setUp(self):
        np.set_printoptions(threshold=10, linewidth=120, precision=6)
        self.ex5data = scipy.io.loadmat("ex5data1.mat")
        X1 = self.ex5data['X']
        tmp = np.ones((X1.shape[0], 1))
        self.ext_X = np.concatenate((tmp, X1), 1)


# theta = [1 ; 1];
# [J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);


    def test_part2(self):
        X = self.ext_X
        m,n = X.shape

        y = self.ex5data['y']
        theta = np.ones(n)
        lamda = 1.0
        J = regularized_linear.computeCostFunction(theta, X, y, lamda)
        self.assertAlmostEqual(J, 303.993, places=2)

    def test_part3(self):
        X = self.ext_X
        m,n = X.shape

        y = self.ex5data['y']
        theta = np.ones(n)
        lamda = 1.0
        grad = regularized_linear.computeGradient(theta, X, y, lamda)
        self.assertIsInstance(grad, np.matrix)
        self.assertEqual(grad.shape, (1, n))
        self.assertAlmostEqual(grad[0,0], -15.30, places=1)
        self.assertAlmostEqual(grad[0,1], 598.250, places=1)


        # self.assertEqual(mat[4,4], 1)
        # self.assertEqual(mat[0,1], 0)


    # def test_part3(self):
    #     X, y = linear_regression.add_column_of_ones_to_X(self.ex1data1)
    #     m,n = self.ex1data1.shape
    #     theta = np.zeros((n, 1))
    #     J = linear_regression.computeCost(X, y, theta)
    #     self.assertAlmostEqual(J, 32.07, places=2)



if __name__ == '__main__':
    unittest.main()
