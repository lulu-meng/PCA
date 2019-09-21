#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.decomposition import pca
#from sklearn.decomposition import TSNE, TruncatedSVD

def apply_pca(data):
    K=2
    model = pca.PCA(n_components=K).fit(data)
    data_new = model.transform(data)
    return data_new

def apply_svd(data):
    #to do
    #use sklearn svd
    return data
    
def apply_tsne(data):
    #to do
    #use sklearn svd
    return data