import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import sqrt
from scipy.cluster import vq
from scipy.spatial.distance import cdist
from sklearn.datasets.samples_generator import make_blobs
#import pandas as pd

def initialize_parameters(parameters=None):
    """Auxiliar function to set default values to all the parameters not
    given a value by the user.

    """
    parameters = {} if not parameters else parameters

    def safe_pull_value(parameters, key, default):
        return parameters.get(key, default)

    # number of clusters desired
    K = safe_pull_value(parameters, 'K', 6)

    # maximum number of iterations
    I = safe_pull_value(parameters, 'I', 100)

    # maximum of number of pairs of clusters which can be merged
    P = safe_pull_value(parameters, 'P', 4)

    # threshold value for  minimum number of samples in each cluster
    # (discarding clusters)
    THETA_M = safe_pull_value(parameters, 'THETA_M', 100)

    # threshold value for standard deviation (for split)
    THETA_S = safe_pull_value(parameters, 'THETA_S', 1)
    # threshold value for pairwise distances (for merge)
    THETA_C = safe_pull_value(parameters, 'THETA_C', 20)

    # percentage of change in clusters between each iteration
    # (to stop algorithm)
    THETA_O = 0.05

    # can use any of both fixed or random
    # number of starting clusters
    # k = np.random.randint(1, K)
    k = safe_pull_value(parameters, 'k', K)

    ret = locals()
    ret.pop('safe_pull_value')
    ret.pop('parameters')
    globals().update(ret)


def initial_clusters(img, k):
    """
    Define initial clusters centers as startup.
    """
    np.random.shuffle(img)
    centers = img[0:k]

    return centers

def discard_clusters(img_class, centers, clusters_list):
    """
        Discard clusters with fewer than THETA_M.
        """
    k = centers.shape[0]
    to_delete = np.array([])
    for cluster in range(0, k):
        indices = np.where(img_class == clusters_list[cluster])[0]
        total_per_cluster = indices.size
        if total_per_cluster <= THETA_M:
            to_delete = np.append(to_delete, cluster)
    if to_delete.size:
        new_centers = np.delete(centers, to_delete, axis=0)
        new_clusters_list = np.delete(clusters_list, to_delete)
    else:
        new_centers = centers
        new_clusters_list = clusters_list
    return new_centers, new_clusters_list

def update_clusters(img, img_class, centers, clusters_list):
    cluster = centers.shape[0]
    for i in range(cluster):
        indices = np.where(img_class == clusters_list[i])[0]
        cluster_values = img[indices]
        centers[i] = tf.reduce_mean(cluster_values, axis=0)
    return centers

def split_clusters(img, img_class, centers, clusters_list):
    cluster = centers.shape[0]
    varicance = []
    #sigma = 10
    for i in clusters_list:
        indices = np.where(img_class == clusters_list[i])[0]
        cluster_values = img[indices]
        number = indices.size
        varicance = np.var(cluster_values, axis=0)
        var_max = varicance[np.argmax(varicance)]
        if var_max > THETA_S and number > 2*THETA_M:
            clusters_list = np.append(clusters_list, clusters_list.size)
            a = np.zeros(centers.shape[1])
            b = a + var_max
            centers[i] = centers[i] - b
            centers = np.append(centers, [centers[i] + b], axis=0)
    return centers, clusters_list

def merge_clusters(img_class, centers, clusters_list):
    pair_dists = []
    size = centers.shape[1]
    for i in range(0, size):
        for j in range(0, size):
            if i > j:
                d = sqrt(cdist([centers[i]], [centers[j]], metric= 'sqeuclidean'))
                pair_dists.append((d, (i, j)))
    first_p_elements = pair_dists[:P]
    below_threshold = [(c1, c2) for d, (c1, c2) in first_p_elements if d < THETA_C]
    if below_threshold:
        k = centers.shape[1]
        count_per_cluster = np.zeros(k)
        to_add = np.array([])  # new clusters to add
        to_delete = np.array([])  # clusters to delete
        for cluster in range(0, k):
            result = np.where(img_class == clusters_list[cluster])
            indices = result[0]
            count_per_cluster[cluster] = indices.size

        for c1, c2 in below_threshold:
            c1_count = float(count_per_cluster[c1])
            c2_count = float(count_per_cluster[c2])
            factor = 1.0 / (c1_count + c2_count)
            weight_c1 = c1_count * centers[c1]
            weight_c2 = c2_count * centers[c2]

            value = factor * (weight_c1 + weight_c2)

            to_add = np.append(to_add, value)
            to_delete = np.append(to_delete, [c1, c2])

            # delete old clusters and their indices from the availables array
            centers = np.delete(centers, to_delete, axis=0)
            clusters_list = np.delete(clusters_list, to_delete)

            # generate new indices for the new clusters
            # starting from the max index 'to_add.size' times
            start = int(clusters_list.max())
            end = to_add.size + start

            centers = np.append(centers, to_add, axis=0)
            clusters_list = np.append(clusters_list, range(start, end))

            #centers, clusters_list = sort_arrays_by_first(centers, clusters_list)
    return centers, clusters_list


def isodata(img, parameters=None):
    global K, I, P, THETA_M, THETA_S, THETA_C, THETA_O, k
    initialize_parameters(parameters)
    #clusters_list = np.arange(k)  # number of clusters availables
    centers = initial_clusters(img, k)
    for iter in range(0, I):
        #last_centers = centers.copy()
        # assing each of the samples to the closest cluster center
        k = centers.shape[0]
        clusters_list = np.arange(k)
        img_class, dists = vq.vq(img, centers)
        centers, clusters_list = discard_clusters(img_class, centers, clusters_list)
        k = centers.shape[0]
        clusters_list = np.arange(k)
        img_class, dists = vq.vq(img, centers)
        centers = update_clusters(img, img_class, centers, clusters_list)
        k = centers.shape[0]
        if k <= (K / 2.0):
            centers, clusters_list = split_clusters(img, img_class,centers, clusters_list)
        elif k > (K * 2.0):
            centers, clusters_list = merge_clusters(img_class, centers,clusters_list)
        else:
            pass
        img_class, dists = vq.vq(img, centers)


    return img_class
