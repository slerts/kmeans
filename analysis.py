#! /usr/bin/env python
'''
Project Name: Desert Omega
Description: Data analysis for Assignment 2 of K-means Algorithm for COSC 4P67
Author: ns12oj at brocku dot ca
Tested against S-sets obtained from https://cs.joensuu.fi/sipu/datasets/

run from base dir -- python src/kmeans.py
data is located in data/
'''
import sys
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
import kmeans

K_VALUE = [4, 8, 15, 21, 30]
DATA_FILES = ["data/s1.txt", "data/s2.txt", "data/s3.txt", "data/s4.txt"]
CB_FILES = ["data/s1-cb.txt", "data/s2-cb.txt", "data/s3-cb.txt", "data/s4-cb.txt"]
DELIMITER = ","
DIST_MEASURE = ["euclidian", "chebyshev", "manhatten"]


# for index, input_file in enumerate(DATA_FILES):
#     print(input_file)
#     for dist in DIST_MEASURE:
#         results = []
#         print(dist)
#         for k in K_VALUE:
#             print(k)
#             desert_omega = kmeans.KMeans()
#             dpts, cnts, dind = desert_omega.run(input_file, None, DELIMITER, k, dist)
#             data_points = pd.DataFrame.from_records(dpts, columns=['x', 'y', 'class'])
#             cluster_points = pd.DataFrame.from_records(cnts, columns=['x', 'y'])
#             point_cmap = pyplot.get_cmap('Set3')
#             centroid_cmap = pyplot.get_cmap('Reds')
#             plot_title =  str(index) + '-' + dist + '-' + str(k) + '.png'
#             ax = data_points.plot.scatter(x='x', y='y', c=data_points['class'], colormap=point_cmap, colorbar=False, fontsize=10, title=plot_title)
#             cluster_points.plot.scatter(x='x', y='y', marker='^', colormap='Reds', s=50,colorbar=False, ax=ax)
#             ax.set_aspect('equal')
#             #pyplot.show()
#             pyplot.savefig(plot_title, dpi=300, bbox_inches='tight')
#             pyplot.close()
#             results.append([input_file, dist, str(k), str(dind)])
#         data = pd.DataFrame.from_records(results, columns=['input_file', 'distance_measure', 'k_value', 'dunn_index'])
#         results_file = str(index) + '-' + dist + '-' + str(k) + '-results.csv'
#         data.to_csv(results_file)


for index, (input_file, cb_file) in enumerate(zip(DATA_FILES, CB_FILES)):
    print(input_file)
    results = []
    for dist in DIST_MEASURE:
        print(dist)
        k = 15
        desert_omega = kmeans.KMeans()
        dpts, cnts, dind = desert_omega.run(input_file, cb_file, DELIMITER, k, dist)
        data_points = pd.DataFrame.from_records(dpts, columns=['x', 'y', 'class'])
        cluster_points = pd.DataFrame.from_records(cnts, columns=['x', 'y'])
        point_cmap = pyplot.get_cmap('Set3')
        centroid_cmap = pyplot.get_cmap('Reds')
        plot_title =  str(index) + '-cb-' + dist + '-' + str(k) + '.png'
        ax = data_points.plot.scatter(x='x', y='y', c=data_points['class'], colormap=point_cmap, colorbar=False, fontsize=10, title=plot_title)
        cluster_points.plot.scatter(x='x', y='y', marker='^', colormap='Reds', s=50,colorbar=False, ax=ax)
        ax.set_aspect('equal')
        #pyplot.show()
        pyplot.savefig(plot_title, dpi=300, bbox_inches='tight')
        results.append([input_file, dist, str(k), str(dind)])
    data = pd.DataFrame.from_records(results, columns=['input_file', 'distance_measure', 'k_value', 'dunn_index'])
    results_file = str(index) + '-cb_results.csv'
data.to_csv(results_file)
