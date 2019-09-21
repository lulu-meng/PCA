#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.decomposition import pca
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def apply_pca(data):
    K=2
    model = pca.PCA(n_components=K).fit(data)
    data_new = model.transform(data)
    return data_new

def apply_svd(data):
    #use sklearn svd
    k = 2
    #get components
    #components = TruncatedSVD(k).fit(data).components_
    return TruncatedSVD(k).fit_transform(data)

    
def apply_tsne(data):
    #use sklearn svd
    k = 2
    # get embeddings
    #embeddings = TSNE(k).fit(data).embedding_
    return TSNE(k).fit_transform(data)
