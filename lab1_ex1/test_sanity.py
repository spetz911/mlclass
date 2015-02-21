import unittest
import linear_regression
import numpy as np


class LinearRegression(unittest.TestCase):

    def setUp(self):
        self.ex1data1 = np.genfromtxt("ex1data1.txt", delimiter=',')


    def test_part1(self):
        mat = linear_regression.warmUpExercise()
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (5,5))
        self.assertEqual(mat[4,4], 1)
        self.assertEqual(mat[0,1], 0)


    def test_part3(self):
        X, y = linear_regression.add_column_of_ones_to_X(self.ex1data1)
        m,n = self.ex1data1.shape
        theta = np.zeros((n, 1))
        J = linear_regression.computeCost(X, y, theta)
        self.assertAlmostEqual(J, 32.07, places=2)



if __name__ == '__main__':
    unittest.main()
