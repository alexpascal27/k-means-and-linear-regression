import numpy as np
import math


class LinearRegression:
    def __init__(self):
        self.m = 1
        self.c = 0

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Need to work out alpha, beta, gamma and delta. After we work these out we can find m and c
        :param X: 2D numpy.ndarray
        :param y: 1D numpy.ndarray
        :return: Nothing
        """
        # Loop through x and y at the same time and add to alpha, beta, gamma and delta as we traverse
        alpha = 0
        beta = 0
        gamma = 0
        delta = 0
        n = len(X)

        for i in range(n):
            x_i = X[i][0]
            y_i = y[i]
            alpha += (y_i * x_i)
            beta += math.pow(x_i, 2)
            gamma += x_i
            delta += y_i

        # use beta, gamma, alpha, delta to work out the m and c
        beta_gamma_n_matrix = np.array([[beta, gamma], [gamma, n]])
        alpha_delta_matrix = np.array([[alpha], [delta]])
        m_c_matrix = np.matmul(np.linalg.inv(beta_gamma_n_matrix), alpha_delta_matrix)
        self.m = m_c_matrix[0][0]
        self.c = m_c_matrix[1][0]

    def predict(self, X: np.array) -> np.array:
        """
        Using the m and c values calculated from the fit function to predict the y value for the input X
        :param X: 1D numpy.ndarray
        :return: 1D numpy.ndarray
        """
        # Init the array of y values
        y = np.array([])
        # loop through all the x values and use m and c to work out the respective y value
        for i in range(len(X)):
            y_value = self.m * X[i] + self.c
            y = np.append(y, [y_value])
        return y

