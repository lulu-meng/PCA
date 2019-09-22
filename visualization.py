#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def visualization(data,labels,title,axis):
    #draw the data points with a scatter plot, and color them according to their labels

    labels_list = np.unique(labels)

    fig, ax = plt.subplots()

    for label in labels_list:
        ix = np.where(np.array(labels) == label)
        ax.scatter(data[ix,0],data[ix,1],label=label,s=15,alpha=0.5)
    ax.legend()
    plt.title(title)
    plt.xlabel(axis + ' 1')
    plt.ylabel(axis + ' 2')
    plt.show()
    return
