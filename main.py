#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import numpy as np
import visualization as vs
import sklearn_apply as skap
import pca as mypca

def read(file_path, n):
    file = open(file_path, "r")
    content = file.readlines()
    file.close()
    
    x = []
    y = []
    rows = len(content)
    for line in content:
        line_list = re.split(r'\t+', line.rstrip('\n'))
        x.append(line_list[0:n-1])
        y.append(line_list[n-1])
        
    datamat=np.zeros((rows,n-1))#初始化矩阵
    for i in range(rows):
        datamat[i,:] = x[i][:]
        
    print("x:",x)
    print("y:",y)
    print("mat:",datamat)

    return datamat,y


def main():
    data, labels = read("data/Homework2_pca_a.txt",5)
    data = data.astype(np.float)
    
    my_pca_res = mypca.pca(data)
    sklearn_pca_res = skap.apply_pca(data)
    
    vs.visualization(my_pca_res,labels)
    vs.visualization(sklearn_pca_res,labels)

if __name__ == "__main__":
    main()