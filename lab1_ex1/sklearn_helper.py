#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""

# Code source: Jaques Grobler
# License: BSD 3 clause


from matplotlib import pyplot
import numpy as np
import sklearn.datasets
import sklearn.linear_model

def main():
    # Load the diabetes dataset
    diabetes = sklearn.datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis]
    diabetes_X_temp = diabetes_X[:, :, 2]

    print(diabetes_X_temp[:10])

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X_temp[:-20]
    diabetes_X_test = diabetes_X_temp[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    print(diabetes_y_train[:10])

    # exit(0)

    # Create linear regression object
    regr = sklearn.linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    # Plot outputs
    pyplot.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    pyplot.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
             linewidth=3)

    pyplot.xticks(())
    pyplot.yticks(())

    pyplot.show()









main()


