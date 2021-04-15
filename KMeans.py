import numpy as np
import random
from sklearn.metrics import pairwise_distances_argmin


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
        :param data: 2D ndarray containing our data
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
            arrays_equivalent = self.__are_centroid_centers_equal(self.cluster_centers_, self.old_cluster_centers, error_tolerance=0.01)
            if arrays_equivalent:
                break
            # If not, set the old cluster centers to the current centers and repeat the process
            else:
                self.old_cluster_centers = np.copy(self.cluster_centers_)

    def determine_random_cluster_centers(self, data: np.array) -> None:
        # Reorder the array randomly
        reordered_data = np.random.RandomState(self.n_clusters).permutation(len(data))
        # Pick the first n elements
        list_of_indexes = [reordered_data[i] for i in range(self.n_clusters)]
        # Set those elements to self.cluster_centers_
        self.cluster_centers_ = np.copy(data[list_of_indexes]).astype(float)

    def __update_centroids(self, data: np.array) -> None:
        # initialise the array holding the current sum
        sum_of_data_points_per_cluster = np.zeros((self.n_clusters, len(data[0])), dtype=int)
        # initialise the array that holds the number of points per cluster
        number_of_points_per_cluster = np.zeros(self.n_clusters, dtype=int)

        # sum up the points in cluster and keep track of how many there are in each cluster
        for i in range(len(self.labels_)):
            cluster_index = self.labels_[i]
            sum_of_data_points_per_cluster[cluster_index] += data[i]
            number_of_points_per_cluster[cluster_index] += 1

        # loop through the clusters centers
        for i in range(len(self.cluster_centers_)):
            self.cluster_centers_[i] = sum_of_data_points_per_cluster[i] / number_of_points_per_cluster[i]

    def __are_centroid_centers_equal(self, center1: np.array, center2: np.array, error_tolerance: float) -> bool:
        # if either is None, not equal
        if center1 is None or center2 is None:
            return False
        # check that each point in one center is equivalent to the respective point in the other center array
        for i in range(len(center1)):
            for j in range(len(center1[0])):
                if not (abs(center1[i][j] - center2[i][j]) <= error_tolerance):
                    return False
        return True
