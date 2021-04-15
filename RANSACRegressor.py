import numpy as np


class RANSACRegressor:
    def __init__(self):
        self.m = 1
        self.c = 0
        self.inlier_mask_ = None

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Loop through and calculate new m and c and inlier masks. At each stage the local variables initialised in __init__
        are updated with the values corresponding to the values that have the fewest outliers (smallest error).
        :param X: 2D numpy.ndarray
        :param y: 1D numpy.ndarray
        :return: Nothing
        """
        n = len(X)
        # initialise the inlier mask
        self.inlier_mask_ = np.zeros(n, dtype=bool)

        # loop through all points
        for i in range(n):
            # get next point, find line between input point and get the line's m and c
            m_i, c_i = self.__given_index_get_line(X, y, i)
            # we use this array to store our current inlier mask to compare to the class inlier mask variable
            current_inlier_mask = np.zeros(n, dtype=bool)

            # using the m and c calculated find the inliers
            for j in range(n):
                x_j = X[j][0]
                y_j = y[j]
                estimated_y = (m_i * x_j + c_i)
                current_inlier_mask[j] = self.__is_inlier(estimated_y, y_j)

            # see if we have less inliers at this iteration compared to previous iterations
            inliers_now = np.count_nonzero(current_inlier_mask)
            inliers_before = np.count_nonzero(self.inlier_mask_)
            if inliers_now > inliers_before:
                # if so then we change the m and c and the inlier mask array
                self.m = m_i
                self.c = c_i
                self.inlier_mask_ = np.copy(current_inlier_mask)

    def __is_inlier(self, estimated_y: float, actual_y: float):
        return abs(estimated_y - actual_y) < 33

    def __given_index_get_line(self, X: np.array, y: np.array, i: int):
        # Find next index
        next_index = self.__find_next_index(i, len(X))
        # Get the x and y values
        x_1 = X[i][0]
        x_2 = X[next_index][0]
        y_1 = y[i]
        y_2 = y[next_index]
        # m = {change in y} / {change in x}
        m_i = (y_2 - y_1) / (x_2 - x_1)
        # c = y_i - m*x_i
        c_i = y_1 - m_i * x_1

        return m_i, c_i

    def __find_next_index(self, current_index: int, length_of_collection: int) -> int:
        # if last index return first index
        if current_index == (length_of_collection - 1):
            return 0
        # otherwise return index + 1
        else:
            return current_index + 1

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
