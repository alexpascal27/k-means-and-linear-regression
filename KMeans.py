import numpy as np
import random
from sklearn.metrics import pairwise_distances_argmin
import math


class KMeans:
    # Storing the cluster centers from the previous iteration to compare
    old_cluster_centers = None

    def __init__(self, n_clusters: int):
        # number of clusters the user wants split the data into
        self.n_clusters = n_clusters
        # np.array that stores the points that are the current centroids
        self.cluster_centers_ = None
        # nd.array that stores the cluster number each data point was assigned to
        self.labels_ = None

    def fit(self, data: np.array) -> None:
        """
        Public Function that deals with the K means fitting.
        :param data:
        :return: we dont return anything, because the user can just access the labels and cluster centers arrays
        """

        # "Create a random centroid for each cluster."
        self.determine_random_cluster_centers(data)
        # Run until the arrays are equivalent
        while True:
            # "For each data point identify the closest centroid and assign it to the corresponding cluster."
            self.labels_ = pairwise_distances_argmin(data, self.cluster_centers_)
            # "Compute a new centroid for each cluster based on the current cluster members"
            self.__update_centroids(data)

            # Check if the centroids from the previous iteration is equivalent to the current iteration (means we can stop iterating)
            arrays_equivalent = self.__are_centroid_centers_equal(self.cluster_centers_, self.old_cluster_centers, 0.1)
            if arrays_equivalent:
                break
            # If not, set the old cluster centers to the current centers and repeat the process
            else:
                self.old_cluster_centers = np.copy(self.cluster_centers_)

    def determine_random_cluster_centers(self, data: np.array) -> None:
        # First initialise the arrays with zeros so we don't get index out of range exception
        self.cluster_centers_ = np.zeros(shape=(self.n_clusters, len(data[0])), dtype=float)
        # Randomise n different points
        for i in range(self.n_clusters):
            # Randomise an index for the center point
            index = random.randint(0, len(data))
            # Get data point at that range and set it to a centroid
            self.cluster_centers_[i] = data[index]

    def __label_data(self, data: np.array) -> np.array:
        new_labels_ = np.zeros(shape=(len(data)), dtype=int)
        # Go through each data point
        for i in range(len(data)):
            # Get the closest cluster to the current point
            index_of_closest_cluster = self.__get_closest_cluster(data[i])
            # Add to labels array
            new_labels_[i] = index_of_closest_cluster

        return new_labels_

    def __get_closest_cluster(self, data_point: np.array) -> int:
        # index of closest cluster
        i = 0
        # smallest distance
        smallest_distance = math.inf
        # loop through cluster centers and see which is the shortest distance away
        for j in range(len(self.cluster_centers_)):
            distance = 0
            # get the distance between the point and the cluster center
            for k in range(len(data_point)):
                distance += abs(self.cluster_centers_[j][k] - data_point[k])
            # update the smallest distance and update the index of closest cluster so far
            if distance < smallest_distance:
                smallest_distance = distance
                i = j
        # after looping through all cluster centers the one that has the smallest distance at the end is the closest so return
        return i

    def __update_centroids(self, data: np.array) -> None:
        # loop through the clusters centers
        for i in range(len(self.cluster_centers_)):
            # get the array of indexes for the current cluster
            array_of_indexes = np.where(self.labels_ == i)[0]
            # get how many points in cluster
            n = len(array_of_indexes)
            # sum up all the points in cluster
            sum_of_points = np.zeros(len(data[0]), dtype=int)
            for j in range(len(array_of_indexes)):
                sum_of_points = np.add(sum_of_points, data[j])
            # assign the new center
            self.cluster_centers_[i] = sum_of_points/n

    def __are_centroid_centers_equal(self, center1: np.array, center2: np.array, error_tolerance: float) -> bool:
        if center1 is None or center2 is None:
            return False

        for i in range(len(center1)):
            for j in range(len(center1[0])):
                if not (abs(center1[i][j] - center2[i][j]) <= error_tolerance):
                    return False
        return True
