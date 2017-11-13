---
layout: post
title: Machine Learning Note(10)
date: 2017-11-13
categories: blog
tags: [note]
description: Machine Learning
---

# SVM

In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called support vector clustering and is often[citation needed] used in industrial applications either when data are not labeled or when only some data are labeled as a preprocessing for a classification pass.[wiki]

Let's make some datas.

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, cross_validation,svm


def load_data_regression():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)


def load_data_classification():
    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target
    return cross_validation.train_test_split(x_train, y_train, test_size=0.25,random_state=0,stratify=y_train)
```


## Linear SVM

```
def test_Lsvm(*data):
    x_train, x_test, y_train, y_test = data
    cls = svm.LinearSVC()
    cls.fit(x_train, y_train)
    print('coefficients:%s, intercept %.s' % (cls.coef_, cls.intercept_))
    print('score : %.2f' % cls.score(x_test, y_test))


x_train, y_train, x_test, y_test = load_data_classification()
test_Lsvm(x_train, y_train, x_test, y_test)

```

It's nearly the best way to do it, but it's too time-consuming.

```
coefficients:[[ 0.21663167  0.38734209 -0.82116846 -0.44244733]
 [-0.14317247 -0.76685858  0.52311647 -1.00256705]
 [-0.80600518 -0.91890408  1.25789796  1.72566837]], intercept 

score : 0.97
```

## Nonlinear classification

```
def test_svcl(*data):
    x_train, x_test, y_train, y_test = data
    cls = svm.SVC(kernel='linear')
    cls.fit(x_train, y_train)
    print('coefficients:%s, intercept %.s' % (cls.coef_, cls.intercept_))
    print('score : %.2f' % cls.score(x_test, y_test))

x_train, y_train, x_test, y_test = load_data_classification()
test_svcl(x_train, y_train, x_test, y_test)
```

The score is one. It is a popular algrithm.

```
coefficients:[[-0.09678346  0.40464239 -0.95269008 -0.50457359]
 [ 0.01882639  0.17448893 -0.54596525 -0.21811023]
 [ 0.487078    0.93180889 -1.77348523 -1.99541406]], intercept 
score : 1.00
```