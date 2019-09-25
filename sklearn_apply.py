#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.decomposition import pca
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np

def apply_pca(data):
    K=2
    data = data.astype(np.float)
    model = pca.PCA(n_components=K).fit(data)
    data_new = model.transform(data)
    return data_new

def apply_svd(data):
    #use sklearn svd
    k = 2
    data = data.astype(np.float)
    #get components
    #components = TruncatedSVD(k).fit(data).components_
    return TruncatedSVD(k).fit_transform(data)

    
def apply_tsne(data):
    #use sklearn svd
    k = 2
    # data = data.astype(np.float)
    #get embeddings
    #embeddings = TSNE(k).fit(data).embedding_
    return TSNE(n_components=k, random_state=1).fit_transform(data)
