#! /usr/bin/env python
'''
Project Name: Desert Omega
Description: K-means Algorithm for COSC 4P67
Author: ns12oj at brocku dot ca
Tested against S-sets obtained from https://cs.joensuu.fi/sipu/datasets/

run from base dir -- python src/kmeans.py
data is located in data/
'''
import sys
import math
import numpy as np

class KMeans(object):
    """KMeans class holds all data structures and functions necessary
    for k-means clustering."""

    def __init__(self):
        self.data_pts = None
        self.centroids = None
        self.cluster_sums = None
        self.centroid_size = None
        self.dunn_index = None

    def load_data(self, data_file, delim):
        """loads data from txt file (requires delimiter)
        tmp for centroid column created and then appended to data pts
        """
        self.data_pts = np.loadtxt(data_file, delimiter = delim, dtype=int)
        cent_list = np.ones((self.data_pts.shape[0], 1), dtype=int) * -1
        self.data_pts = np.hstack((self.data_pts, cent_list))

    def init_centroids(self, k_val, centroid_file, delim):
        """initializes centroids to random points within the data set.
        """
        self.centroid_size = np.zeros(k_val, dtype=int)
        self.cluster_sums = np.zeros((k_val, 2), dtype=int)
        if centroid_file == None:
            self.centroids = np.zeros((k_val, 2), dtype=int)
            for i in xrange(k_val):
                pt = np.random.random_integers(0, self.data_pts.shape[0])
                self.centroids[i] = np.copy(self.data_pts[pt][:2])
        else:
            self.centroids = np.loadtxt(centroid_file, delimiter = delim, dtype=int)

    def distance_measure(self, measure, point, centroid):
        """calculates distance of data point from centroid. 3 distance
        measures offered, needs to be specific via argument. defaults to
        euclidean on bad argument."""
        if measure == "euclidian":
            x_dist = (point[0] - centroid[0]) ** 2
            y_dist = (point[1] - centroid[1]) ** 2
            return int(math.sqrt(x_dist + y_dist))
        elif measure == "chebyshev":
            x_dist = abs(point[0] - centroid[0])
            y_dist = abs(point[1] - centroid[1])
            return max(x_dist, y_dist)
        elif measure == "manhatten":
            x_dist = abs(point[0] - centroid[0])
            y_dist = abs(point[1] - centroid[1])
            return x_dist + y_dist
        return self.distance_measure("euclidian", point, centroid)

    def cluster(self, measure):
        """runs k-means cluster main loop.
        TODO better way to handle loop with numpy functions?
        """
        epochs = 0
        converged = False

        while not converged:
            epochs += 1
            # reset cluster sums and centroid sizes
            self.cluster_sums[:, 0] = 0.0
            self.cluster_sums[:, 1] = 0.0
            self.centroid_size[:] = 0
            # loop through data points list
            for row in self.data_pts:
                # set min distance to max possible
                nearest_centroid = sys.maxint
                # loop through centroid list
                for x in xrange(self.centroids.shape[0]):
                    # calculate distance from data point to centroid
                    dist = self.distance_measure(measure, row, self.centroids[x][:])
                    # if distance is less than current min distance
                    if dist < nearest_centroid:
                        # set the data pt class to the centroid
                        row[2] = x
                        # set the min distance to distance of current centroid
                        nearest_centroid = dist
                # print row[2]
                # increment number in class
                self.centroid_size[int(row[2])] += 1
                # add x, y distance to centroid sums
                self.cluster_sums[row[2]][0] += row[0]
                self.cluster_sums[row[2]][1] += row[1]
            # calculate new centroid positions
            self.cluster_sums = (self.cluster_sums.T / self.centroid_size).T
            # converged if current centroid positions don't change
            if (self.cluster_sums == self.centroids).all():
                converged = True
                break
            # copy new mean locations to mean list
            self.centroids = np.copy(self.cluster_sums)

    #
    def dunn(self, measure):
        """ calculates dunn index for clusters. used as a validity measure
        for gauging the desirability of the clusters."""
        dunn_min = sys.maxint
        dunn_max = -dunn_min - 1
        for outer in self.data_pts:
            for inner in self.data_pts:
                dist = self.distance_measure(measure, outer, inner)
                if outer[2] == inner[2]:
                    if dist > dunn_max:
                        dunn_max = dist
                else:
                    if dist < dunn_min:
                        dunn_min = dist
        self.dunn_index = (dunn_min * 1.0) / (dunn_max * 1.0)


    def run(self, data_file, centroid_file, delim, k_val, measure):
        """execution function to make things happen"""
        self.load_data(data_file, delim)
        self.init_centroids(k_val, centroid_file, delim)
        self.cluster(measure)
        self.dunn(measure)
        return self.data_pts, self.centroids, self.dunn_index