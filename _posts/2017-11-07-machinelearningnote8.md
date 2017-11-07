---
layout: post
title: Machine Learning Note(8)
date: 2017-11-07
categories: blog
tags: [note]
description: Machine Learning
---

# Dimension Reduction

In machine learning and statistics, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration, via obtaining a set of principal variables. It can be divided into feature selection and feature extraction.

For high-dimensional datasets (i.e. with number of dimensions more than 10), dimension reduction is usually performed prior to applying a K-nearest neighbors algorithm (k-NN) in order to avoid the effects of the curse of dimensionality.

Feature extraction and dimension reduction can be combined in one step using principal component analysis (PCA), linear discriminant analysis (LDA), or canonical correlation analysis (CCA) techniques as a pre-processing step followed by clustering by K-NN on feature vectors in reduced-dimension space. In machine learning this process is also called low-dimensional embedding.

For very-high-dimensional datasets (e.g. when performing similarity search on live video streams, DNA data or high-dimensional time series) running a fast approximate K-NN search using locality sensitive hashing, random projection, "sketches"  or other high-dimensional similarity search techniques from the VLDB toolbox might be the only feasible option. [wiki]

## PCA

This is my favorite algrithm. The main linear technique for dimensionality reduction, principal component analysis, performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the low-dimensional representation is maximized. In practice, the covariance (and sometimes the correlation) matrix of the data is constructed and the eigen vectors on this matrix are computed. The eigen vectors that correspond to the largest eigenvalues (the principal components) can now be used to reconstruct a large fraction of the variance of the original data. Moreover, the first few eigen vectors can often be interpreted in terms of the large-scale physical behavior of the system[citation needed]. The original space (with dimension of the number of points) has been reduced (with data loss, but hopefully retaining the most important variance) to the space spanned by a few eigenvectors. 

Today, I use some methods to analyse the iris data.


## PCA

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition,manifold

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target
```

```
def test_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    print('explained variance ratio : %s' % str(pca.explained_variance_ratio_))


x, y = load_data()
test_PCA(x, y)
```

The result is:

```
explained variance ratio : [ 0.92461621  0.05301557  0.01718514  0.00518309]
```

That means the data dimension from 4 4 becomes 2 2

```
def plot_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    x_r = pca.transform(x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 5, 0), (0, 0.5, 0, 5), (0, 5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
        (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()
```

The result shows in the picture:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_8_1.png)

## KPCA

```
def test_Kpca(*data):
    x, y = data
    kernels = ['linear','poly','rbf','sigmoid']
    for kernel in kernels:
        kpca = decomposition.KernelPCA(n_components=None, kernel=kernel)
        kpca.fit(x)
        print('explained lambdas : %s' % str(kpca.lambdas_))
```

The result is big.

```
def plot_kPCA(*data):
    x, y = data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 5, 0), (0, 0.5, 0, 5), (0, 5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
        (0, 0.6, 0.4), (0.5, 0.3, 0.2))

    Params = [(3,1,1),(3,10,1),(3,1,10),(3,10,10),(10,1,1),(10,10,1),(10,1,10),(10,10,10)]
    for i,(p,gamma,r) in enumerate(Params):
        kpc = decomposition.KernelPCA(n_components=2,kernel='poly',gamma=gamma,degree=p,coef0=r)
        kpc.fit(x)
        x_r=kpc.transform(x)
        ax=fig.add_subplot(2,4,i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$(%s(x\cdot z+1)+%s)^{%s}$" % (gamma, r, p))
    plt.suptitle("kpca-poly")
    plt.show()
```

Different kernel make different result.

![result](http://oybqmhgid.bkt.clouddn.com/Figure_8_2.png)

## MDS

```
def test_mds(*data):
    x, y = data
    for n in [4,3,2,1]:
        mds = manifold.MDS(n_components=n)
        mds.fit(x)
        print('stress (n= %d) :%s' % (n,str(mds.stress_)))
```

The results show the smaller the dimensions are the bigger the errors are.

```
stress (n= 4) :11.118898618
stress (n= 3) :17.3622487399
stress (n= 2) :113.526436703
stress (n= 1) :990.940686386
```

```
def plot_mds(*data):
    x, y = data
    pca = manifold.MDS(n_components=2)
    pca.fit(x)
    x_r = pca.fit_transform(x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 5, 0), (0, 0.5, 0, 5), (0, 5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
        (0, 0.6, 0.4), (0.5, 0.3, 0.2))
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()
```

The result is ：

![result](http://oybqmhgid.bkt.clouddn.com/Figure_8_3.png)

## ISOMAP

```
def plot_isomap(*data):
    x, y = data
    ks = [1,5,25,y.size-1]
    fig = plt.figure()
    for i,k in enumerate(ks):
        isomap = manifold.Isomap(n_components=2,n_neighbors=k)
        x_r = isomap.fit_transform(x)
        ax = fig.add_subplot(2,2,i+1)
        colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 5, 0), (0, 0.5, 0, 5), (0, 5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
        (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
       ax.set_xlabel("x[0]")
       ax.set_ylabel("x[1]")
       ax.legend(loc="best")
       ax.set_title("k = %d"%k)
    plt.suptitle("isomap")
    plt.show()
```

The performance looks like kpca.

![result](http://oybqmhgid.bkt.clouddn.com/Figure_8_4.png)


## LLE

```
def plot_lle(*data):
    x, y = data
    ks = [1,5,25,y.size-1]
    fig = plt.figure()
    for i,k in enumerate(ks):
        isomap = manifold.LocallyLinearEmbedding(n_components=2,n_neighbors=k)
        x_r = isomap.fit_transform(x)
        ax = fig.add_subplot(2,2,i+1)
        colors = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 5, 0), (0, 0.5, 0, 5), (0, 5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
        (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1], label="target=%d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title("k = %d"%k)
    plt.suptitle("LLE")
    plt.show()
```


![result](http://oybqmhgid.bkt.clouddn.com/Figure_8_5.png)

It is a good algrithm. The results have low errors.