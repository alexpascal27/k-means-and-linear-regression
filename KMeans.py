import numpy as np
import random
import math


class KMeans:
    # Storing the cluster from the previous iteration to compare
    old_cluster_centers = None
    # Storing a list of the points assigned to each cluster
    points_in_cluster = None

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, data: np.array) -> None:
        self.cluster_centers_ = np.zeros(shape=(self.n_clusters, len(data[0])), dtype=float)
        self.labels_ = np.zeros(shape=(len(data)))
        # "Create a random centroid for each cluster."
        self.determine_cluster_centers(data)

        while True:
            # Initialise the variables needed per loop
            self.init_variables(data)
            # "For each data point identify the closest centroid and assign it to the corresponding cluster."
            self.label_data(data)
            # "Compute a new centroid for each cluster based on the current cluster members"
            self.update_centroids()

            #print("Another One")
            #print(self.old_cluster_centers)
            #()
            #print(self.cluster_centers_)

            arrays_equivalent = self.compare_arrays(self.old_cluster_centers, self.cluster_centers_)
            #print("Arrays Equivalent: " + str(arrays_equivalent))
            if arrays_equivalent:
                break
            else:
                self.old_cluster_centers = np.copy(self.cluster_centers_)

    def init_variables(self, data: np.array) -> None:
        self.points_in_cluster = []
        for i in range(self.n_clusters):
            self.points_in_cluster.append(np.zeros((0, len(data[0])), dtype=int))

    def determine_cluster_centers(self, data: np.array) -> None:
        # Randomise n different points
        for i in range(self.n_clusters):
            # Randomise an index for the center point
            index = random.randint(0, len(data))
            # Get data point at that range
            data_point = data[index]
            """
            # Ensure we get a different center point compared to other cluster center points
            while data_point in cluster_center_list:
                # Randomise an index for the center point
                index = random.randint(0, len(data))
                # Get data point at that range
                data_point = data[index]
            """
            for j in range(len(data_point)):
                self.cluster_centers_[i][j] = data_point[j]

    def label_data(self, data: np.array) -> None:
        for i in range(len(data)):
            index_of_closest_cluster = self.get_closest_cluster(data[i])
            self.labels_[i] = index_of_closest_cluster
            self.points_in_cluster[index_of_closest_cluster] = np.append(self.points_in_cluster[index_of_closest_cluster], [data[i]], axis=0)

    def get_closest_cluster(self, data_point: np.array) -> int:
        # index of closest cluster
        i = 0
        # smallest distance
        smallest_distance = math.inf
        # loop through cluster centers and see which is the shortest distance away
        for j in range(len(self.cluster_centers_)):
            distance = 0
            for k in range(len(data_point)):
                distance += abs(int(self.cluster_centers_[j][k]) - int(data_point[k]))

            if distance < smallest_distance:
                smallest_distance = distance
                i = j
        return i

    def update_centroids(self) -> None:
        for i in range(self.n_clusters):
            sum_of_points_in_cluster = self.points_in_cluster[i].sum(axis=0)
            self.cluster_centers_[i] = sum_of_points_in_cluster/len(self.points_in_cluster[i])

    def find_average_sum(self, i: int) -> float:
        sum_so_far = 0
        for j in range(len(self.points_in_cluster[i])):
            sum_of_point = 0
            data_point = self.points_in_cluster[i][j]
            for k in range(len(data_point)):
                sum_of_point += data_point[k]
            sum_so_far += sum_of_point

        return sum_so_far/len(self.points_in_cluster[i])

    def get_closest_cluster_point(self, i: int, average: float) -> list:
        closest_distance = math.inf
        closest_point = []
        for j in range(len(self.points_in_cluster[i])):
            sum_of_point = 0
            data_point = self.points_in_cluster[i][j]
            for k in range(len(data_point)):
                sum_of_point += data_point[k]

            distance_of_point_to_average = abs(float(sum_of_point) - average)
            if distance_of_point_to_average < closest_distance:
                closest_point = self.points_in_cluster[i][j]

        return closest_point

    def compare_arrays(self, array1: np.array, array2: np.array) -> False:
        if array1 is None or array2 is None:
            if array1 is None and array2 is None:
                return True
            else:
                return False

        for i in range(len(array1)):
            found_in_array2 = False
            for j in range(len(array2)):
                if np.array_equal(array1[i], array2[j]):
                    found_in_array2 = True

            if not found_in_array2:
                return False

        return True
