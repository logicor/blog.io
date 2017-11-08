---
layout: post
title: Machine Learning Note(8)
date: 2017-11-07
categories: blog
tags: [note]
description: Machine Learning
---

# Clustering and EM Algrithm

Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters). It is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression, and computer graphics.

Cluster analysis itself is not one specific algorithm, but the general task to be solved. It can be achieved by various algorithms that differ significantly in their notion of what constitutes a cluster and how to efficiently find them. Popular notions of clusters include groups with small distances among the cluster members, dense areas of the data space, intervals or particular statistical distributions. Clustering can therefore be formulated as a multi-objective optimization problem. The appropriate clustering algorithm and parameter settings (including values such as the distance function to use, a density threshold or the number of expected clusters) depend on the individual data set and intended use of the results. Cluster analysis as such is not an automatic task, but an iterative process of knowledge discovery or interactive multi-objective optimization that involves trial and failure. It is often necessary to modify data preprocessing and model parameters until the result achieves the desired properties.[wiki]

## Clustering

The notion of a "cluster" cannot be precisely defined, which is one of the reasons why there are so many clustering algorithms. There is a common denominator: a group of data objects. However, different researchers employ different cluster models, and for each of these cluster models again different algorithms can be given. The notion of a cluster, as found by different algorithms, varies significantly in its properties. Understanding these "cluster models" is key to understanding the differences between the various algorithms. 

Evaluation (or "validation") of clustering results is as difficult as the clustering itself. Popular approaches involve "internal" evaluation, where the clustering is summarized to a single quality score, "external" evaluation, where the clustering is compared to an existing "ground truth" classification, "manual" evaluation by a human expert, and "indirect" evaluation by evaluating the utility of the clustering in its intended application.

Internal evaluation measures suffer from the problem that they represent functions that themselves can be seen as a clustering objective. For example, one could cluster the data set by the Silhouette coefficient; except that there is no known efficient algorithm for this. By using such an internal measure for evaluation, we rather compare the similarity of the optimization problems, and not necessarily how useful the clustering is.

External evaluation has similar problems: if we have such "ground truth" labels, then we would not need to cluster; and in practical applications we usually do not have such labels. On the other hand, the labels only reflect one possible partitioning of the data set, which does not imply that there does not exist a different, and maybe even better, clustering.[wiki]

For example, data set D can be divided into two set C1 and C2. Introducing sets a , b , c , d , their definition are as follow.

The element in a belongs to C1 and C2 at same time.
The element in b belongs to C1 only.
The element in c belongs to C2 only.
The element are not included by C1 and C2.

EXTERNAL EVALUTION:

JC coefficient:

<img src="http://www.forkosh.com/mathtex.cgi? JC=\frac{a}{a+b+c}">

FM Index:

<img src="http://www.forkosh.com/mathtex.cgi? FMI = \sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}">

As to internal evalution, the distances between classes and within classes are very important.

Here samples are made in GUASS distribution.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture


def creat_data(centers,num=100,std=0.7):
    x, labels_true = make_blobs(n_samples=num,centers=centers,cluster_std=std)
    return x,labels_true
```

Kmeans:

```
def test_kmeans(*data):
    x, labels_true = data
    clst = cluster.KMeans()
    clst.fit(x)
    predict_lables = clst.predict(x)
    print("ARI :%s"%adjusted_rand_score(labels_true,predict_lables))
    print("center distance:%s"%clst.inertia_)


centers = [[1,1],[2,2],[1,2],[10,20]]
x, labels_true = creat_data(centers,1000,0.5)
test_kmeans(x, labels_true )
```

Result:

```
ARI :0.3711901577836934
center distance:248.973504215
```

ARI means the performance of the algrithm.

DBSCAN(Density-Based Spatial Clustering of Applications with Noise):

```
def test_dbscan(*data):
    x, labels_true = data
    clst = cluster.DBSCAN()
    clst.fit(x)
    predict_lables = clst.fit_predict(x)
    print("ARI :%s"%adjusted_rand_score(labels_true,predict_lables))
    print("sample num:%s"%len(clst.core_sample_indices_))

centers = [[1,1],[2,2],[1,2],[10,20]]
x, labels_true = creat_data(centers,1000,0.5)
test_dbscan(x, labels_true )

```

```
ARI :0.3314792246832924
sample num:988
```

The sample is divided into 988 sets.


## Expectation–maximization algorithm

In statistics, an expectation–maximization (EM) algorithm is an iterative method to find maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables. The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step. These parameter-estimates are then used to determine the distribution of the latent variables in the next E step.

The EM algorithm is used to find (locally) maximum likelihood parameters of a statistical model in cases where the equations cannot be solved directly. Typically these models involve latent variables in addition to unknown parameters and known data observations. That is, either missing values exist among the data, or the model can be formulated more simply by assuming the existence of further unobserved data points. For example, a mixture model can be described more simply by assuming that each observed data point has a corresponding unobserved data point, or latent variable, specifying the mixture component to which each data point belongs.

Finding a maximum likelihood solution typically requires taking the derivatives of the likelihood function with respect to all the unknown values, the parameters and the latent variables, and simultaneously solving the resulting equations. In statistical models with latent variables, this is usually impossible. Instead, the result is typically a set of interlocking equations in which the solution to the parameters requires the values of the latent variables and vice versa, but substituting one set of equations into the other produces an unsolvable equation.

The EM algorithm proceeds from the observation that the following is a way to solve these two sets of equations numerically. One can simply pick arbitrary values for one of the two sets of unknowns, use them to estimate the second set, then use these new values to find a better estimate of the first set, and then keep alternating between the two until the resulting values both converge to fixed points. It's not obvious that this will work at all, but it can be proven that in this context it does, and that the derivative of the likelihood is (arbitrarily close to) zero at that point, which in turn means that the point is either a maximum or a saddle point. In general, multiple maxima may occur, with no guarantee that the global maximum will be found. Some likelihoods also have singularities in them, i.e., nonsensical maxima. For example, one of the solutions that may be found by EM in a mixture model involves setting one of the components to have zero variance and the mean parameter for the same component to be equal to one of the data points.[wiki]

