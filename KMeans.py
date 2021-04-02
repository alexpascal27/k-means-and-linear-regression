import numpy as np
import random
import math


class KMeans:
    # Storing the cluster from the previous iteration to compare
    old_cluster_centers = None

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

        # Storing the sum of values so that mid point can be determined
        self.sum_points_in_cluster = None
        self.occurrences_of_points_in_cluster = None

    def fit(self, data: np.array) -> None:
        self.init_variables(data)
        need_to_iterate = False
        print(len(data))
        # "Create a random centroid for each cluster."
        self.determine_cluster_centers(data)
        # "For each data point identify the closest centroid and assign it to the corresponding cluster."
        self.label_data(data)
        # "Compute a new centroid for each cluster based on the current cluster members"
        self.update_centroids()
        # "Loop back to step 3 until the assignment of clusters is stable"

        print("Another One")
        #print(self.old_cluster_centers)
        #print()
        #print(self.cluster_centers_)

        if self.old_cluster_centers is None:
            need_to_iterate = True
        else:
            if not np.array_equal(self.old_cluster_centers, self.cluster_centers_):
                need_to_iterate = True

        if need_to_iterate:
            self.old_cluster_centers = np.copy(self.cluster_centers_)
            self.fit(data)

    def init_variables(self, data: np.array) -> None:
        if self.cluster_centers_ is None:
            self.cluster_centers_ = np.zeros(shape=(self.n_clusters, len(data[0])), dtype=int)

        if self.labels_ is None:
            self.labels_ = np.zeros(shape=(len(data)))

        self.occurrences_of_points_in_cluster = np.zeros(shape=self.n_clusters, dtype=int)
        self.sum_points_in_cluster = np.zeros(shape=(self.n_clusters, len(data[0])), dtype=int)

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

        # Assign the list we generated to the cluster_centers variable

    def label_data(self, data: np.array) -> None:
        for i in range(len(data)):
            index_of_closest_cluster = self.get_closest_cluster(data[i])
            self.labels_[i] = index_of_closest_cluster
            """
            if i < 20:
        
            print("Data point in cluster with index: " + str(index_of_closest_cluster))
            print("Before Sum and occurrences")
            print(self.sum_points_in_cluster)
            print("===")
            print(self.occurrences_of_points_in_cluster)
            print("-----")
            # Add to sum for that index
            print("Adding")
            print(self.sum_points_in_cluster[index_of_closest_cluster])
            print("And")
            print(data[i])
            
            """
            self.sum_points_in_cluster[index_of_closest_cluster] = np.add(self.sum_points_in_cluster[index_of_closest_cluster], data[i])
            # Add to occurrence number for that index
            self.occurrences_of_points_in_cluster[index_of_closest_cluster] += 1

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
        """
        print("--------------")
        print(sum_in_cluster)
        print()
        print(data_points_in_cluster)
        print("--------------")
        """
        for i in range(len(self.sum_points_in_cluster)):
            self.cluster_centers_[i] = self.sum_points_in_cluster[i] / self.occurrences_of_points_in_cluster[i]

    # Returns true if lists are the same
    def compare_list(self, list1: list, list2: list) -> bool:
        if len(list1) != len(list2):
            return False

        for i in range(len(list1)):
            if type(list1[i]) is np.ndarray:
                if not self.compare_list(list1[i], list2[i]):
                    return False
            else:
                if list1[i] != list2[i]:
                    return False

        return True
