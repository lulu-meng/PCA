#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def sort_key(x):
    return x[0]

def pca(data):
    components = 2
    rows, cols = data.shape
    #substract the mean
    normal_data = data
    for i in range(cols):
        mean = np.mean(data[:,i])
        for j in range(rows):
            normal_data[j][i] -= mean
    #calculate the covariance matrix
    cov_mat = np.cov(np.transpose(normal_data))
    #calculate the eigenvectors and eigenvalues
    eigenval, eigenvec = np.linalg.eig(cov_mat)
    #sorting eigenvectors by correlating eigenvalue
    tmp_pairs = [(np.abs(eigenval[i]), eigenvec[:,i]) for i in range(cols)]
    tmp_pairs.sort(key=sort_key, reverse=True)
    #forming a feature vector
    feature = np.array([ele[1] for ele in tmp_pairs[:components]])
    feature = np.transpose(feature)
    #new dataset
    result = normal_data @ feature
    return result
