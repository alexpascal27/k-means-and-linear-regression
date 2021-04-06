import numpy as np
import random
import math


class KMeans:
    # Storing the cluster from the previous iteration to compare
    old_cluster_centers = None
    # Storing a list of the points assigned to each cluster
    points_in_cluster = None

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

        # First initialise the arrays with zeros so we don't get index out of range exception
        self.cluster_centers_ = np.zeros(shape=(self.n_clusters, len(data[0])), dtype=float)
        self.labels_ = np.zeros(shape=(len(data)))
        # "Create a random centroid for each cluster."
        self.__determine_random_cluster_centers(data)

        # Run until the arrays are equivalent
        while True:
            # Initialise the variables needed per loop
            self.__init_variables(data)
            # "For each data point identify the closest centroid and assign it to the corresponding cluster."
            self.__label_data(data)
            # "Compute a new centroid for each cluster based on the current cluster members"
            self.__update_centroids()

            # Check if the centroids from the previous iteration is equivalent to the current iteration (means we can stop iterating)
            arrays_equivalent = np.array_equal(self.old_cluster_centers, self.cluster_centers_)
            if arrays_equivalent:
                break
            # If not, set the old cluster centers to the current centers and repeat the process
            else:
                self.old_cluster_centers = np.copy(self.cluster_centers_)

    def __init_variables(self, data: np.array) -> None:
        # Here we set the points in cluster to an empty array according to the data
        self.points_in_cluster = []
        for i in range(self.n_clusters):
            self.points_in_cluster.append(np.zeros((0, len(data[0])), dtype=int))

    def __determine_random_cluster_centers(self, data: np.array) -> None:
        # Randomise n different points
        for i in range(self.n_clusters):
            # Randomise an index for the center point
            index = random.randint(0, len(data))
            # Get data point at that range and set it to a centroid
            self.cluster_centers_[i] = data[index]

    def __label_data(self, data: np.array) -> None:
        # Go through each data point
        for i in range(len(data)):
            # Get the closest cluster to the current point
            index_of_closest_cluster = self.__get_closest_cluster(data[i])
            # Add to labels array
            self.labels_[i] = index_of_closest_cluster
            # Add the point to the cluster specific array (used later for computing the new centroids)
            self.points_in_cluster[index_of_closest_cluster] = np.append(self.points_in_cluster[index_of_closest_cluster], [data[i]], axis=0)

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
                distance += abs(int(self.cluster_centers_[j][k]) - int(data_point[k]))
            # update the smallest distance and update the index of closest cluster so far
            if distance < smallest_distance:
                smallest_distance = distance
                i = j
        # after looping through all cluster centers the one that has the smallest distance at the end is the closest so return
        return i

    def __update_centroids(self) -> None:
        # loop through the clusters centers
        for i in range(len(self.cluster_centers_)):
            # find the sum of columns of the data points
            sum_of_points_in_cluster = self.points_in_cluster[i].sum(axis=0)
            # then divide by how many points there are - to find the average, then assign the average to the cluster centers array element
            self.cluster_centers_[i] = sum_of_points_in_cluster/len(self.points_in_cluster[i])
