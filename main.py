#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import numpy as np
import visualization as vs
import sklearn_apply as skap
import pca as mypca
import matplotlib.pyplot as plt
import pandas as pd

def read(file_path, n):
    file = open(file_path, "r")
    content = file.readlines()
    file.close()
    data = []
    labels = []
    rows = len(content)
    for line in content:
        line_list = re.split(r'\t+', line.rstrip('\n'))
        data.append(line_list[0:n-1])
        labels.append(line_list[n-1])

    data = np.asarray(data)
    return data, labels


def main():
    # change include read number of columns
    data, labels = read("Homework2_pca_c.txt", 12)
    # data = data.astype(np.float)

    my_pca_res = mypca.pca(data)
    sklearn_pca_res = skap.apply_pca(data)
    sklearn_svd_res = skap.apply_svd(data)
    sklearn_tsne_res = skap.apply_tsne(data)


    vs.visualization(my_pca_res,labels,'my_pca','PC')
    vs.visualization(sklearn_pca_res,labels,'sklearn_pca','PC')
    vs.visualization(sklearn_svd_res,labels,'sklearn_svd','SV')
    vs.visualization(sklearn_tsne_res,labels,'sklearn_tsne','tSNE')

if __name__ == "__main__":
    main()
