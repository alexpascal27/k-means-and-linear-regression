import numpy as np
import random
from sklearn.metrics import pairwise_distances_argmin


class KMeans:
    # Storing the cluster centers from the previous iteration to compare
    old_cluster_centers = None
    # Storing the labels from the previous iteration to compare
    old_labels = None

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
        #print("DataLen: " + str(len(data)))
        # Run until the arrays are equivalent
        while True:
            # "For each data point identify the closest centroid and assign it to the corresponding cluster."
            self.labels_ = pairwise_distances_argmin(data, self.cluster_centers_)
            # "Compute a new centroid for each cluster based on the current cluster members"
            self.__update_centroids(data)

            print("---------")
            print(self.old_labels)
            print(self.labels_)
            print("---------")

            # Check if the centroids from the previous iteration is equivalent to the current iteration (means we can stop iterating)
            arrays_equivalent = np.array_equal(self.old_labels, self.labels_)
            if arrays_equivalent:
                break
            # If not, set the old cluster centers to the current centers and repeat the process
            else:
                #self.old_cluster_centers = np.copy(self.cluster_centers_)
                self.old_labels = np.copy(self.labels_)

    def determine_random_cluster_centers(self, data: np.array) -> None:
        # First initialise the arrays with zeros so we don't get index out of range exception
        self.cluster_centers_ = np.zeros(shape=(self.n_clusters, len(data[0])), dtype=float)
        # Randomise n different points
        for i in range(self.n_clusters):
            # Randomise an index for the center point
            index = random.randint(0, len(data))
            # Get data point at that range and set it to a centroid
            self.cluster_centers_[i] = data[index]

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
