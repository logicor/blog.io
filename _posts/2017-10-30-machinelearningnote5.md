---
layout: post
title: Machine Learning Note(5)
date: 2017-10-27
categories: blog
tags: [note]
description: Machine Learning
---

# Decision Tree

A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.(WIKI)


## Principle of Decision Tree

Decision Tree is a greedy algrithm. The sample space is divided into two parts recursively until there is no different data in same part.

 Three steps of constructing decision tree :
 --Feather Selection
 	Commonly used method : variance, Gini coefficient, entropy

E.G.
	Entropy is usually used in classification.
	Suppose that the training set D whose volumn is N has K categories and the volumn of c(k) is |c(k)|. 
	Feature A can divide set D into N sets(D1,D2....). Di set has Ni elements.
	Each set(Di) can be divided into K set. The volumn of Dik is Nik
	The empirical entropy is :
	<img src="http://www.forkosh.com/mathtex.cgi? H(D)=-\sum_{K}^{K=1}\frac{\left | C_{k} \right |}{N}log\frac{\left | C_{k} \right |}{N}">
	The empirical conditional entropy is:
	<img src="http://www.forkosh.com/mathtex.cgi? H(D/A)=\sum \frac{N_{i}}{N}\sum -\left ( \frac{N_{ik}}{N_{i}} \right )log\frac{N_{ik}}{N_{i}} ">
	The information gain can be calculated by:
	<img src="http://www.forkosh.com/mathtex.cgi? g(D,A)=H(D)-H(D/A)">
	The information gain means the decrease of chaos. The more the better.

 --Tree Construction
 	Commonly used method : ID3, C4.5

 E.G.
 	The ID3 algorithm begins with the original set S as the root node. On each iteration of the algorithm, it iterates through every unused attribute of the set and calculates the entropy H ( S ) (or information gain I G ( S )of that attribute. It then selects the attribute which has the smallest entropy (or largest information gain) value. The set S is then split by the selected attribute (e.g. age is less than 50, age is between 50 and 100, age is greater than 100) to produce subsets of the data. The algorithm continues to recurse on each subset, considering only attributes never selected before.
 	--Calculate the entropy of every attribute using the data set S {\displaystyle S} S
 	--Split the set S  into subsets using the attribute for which the resulting entropy (after splitting) is minimum (or, equivalently, information gain is maximum)
 	--Make a decision tree node containing that attribute
 	--Recurse on subsets using remaining attributes.

 --Pruning Tree
 	In order to reduce overfitting, prning tree is necessary. Loss function is always used to balance the fitting degree.


```
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
import matplotlib.pyplot as plt

def creat_data(n):
    np.random.seed(0)
    x = 5* np.random.rand(n, 1)
    y = np.sin(x).ravel()
    noise = (int)(n/5)
    y[::5] += 3*(0.5 - np.random.rand(noise))
    return cross_validation.train_test_split(x, y, test_size=0.5, random_state=1)

def test_DTRegressor(*data):
    x_train, x_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(x_train, y_train)
    print('train score:%f' % (regr.score(x_train, y_train)))
    print('test score: %f' %(regr.score(x_test, y_test)))
    #fig
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.arange(0, 5, 0.01)[:, np.newaxis]
    y = regr.predict(x)
    ax.scatter(x_train, y_train, label = "train sample", c='g')
    ax.scatter(x_test, y_test, label = "test sample", c='r')
    ax.plot(x, y, label = "predict", linewidth=2,alpha = 0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha = 0.5)
    plt.show()

x_train, x_test, y_train, y_test = creat_data(100)
test_DTRegressor(x_train, x_test, y_train, y_test)
```

![result](http://oybqmhgid.bkt.clouddn.com/Figure_2_1.png)

It fits well to the train set, however, it seems badly perform to the test set.
Then we try to figure out the influence of the parameters.

Spliter:

```
def test_DTR_splitter(*data):
    x_train, x_test, y_train, y_test = data
    splitters = ['best', 'random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(x_train, y_train)
        print("splitter%s" % splitter)
        print('train score:%f' % (regr.score(x_train, y_train)))
        print('test score: %f' % (regr.score(x_test, y_test)))

x_train, x_test, y_train, y_test = creat_data(100)
test_DTR_splitter(x_train, x_test, y_train, y_test)
```

The result is :
```
splitterbest
train score:1.000000

test score: 0.535700
splitterrandom

train score:1.000000
Ran 0 tests in 0.000s
test score: 0.511843
```

As to the issue, their work efficiency is the same.

Depth:

```
def test_DTR_depth(*data,maxdepth):
    x_train, x_test, y_train, y_test = data
    depths = np.arange(1, maxdepth)
    traingscores = []
    testingscores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(x_train, y_train)
        traingscores.append(regr.score(x_train, y_train))
        testingscores.append(regr.score(x_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, traingscores, label="traing scores")
    ax.plot(depths, testingscores, label="testing scores")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha = 0.5)
    plt.show()
x_train, x_test, y_train, y_test = creat_data(100)
test_DTR_depth(x_train, x_test, y_train, y_test,maxdepth=20)
```

![result](http://oybqmhgid.bkt.clouddn.com/Figure_2_2.png)

With the increase of the depth, the scores raise to 1. The volumn of set is 100. So the score stops raising when depth is bigger than log2(100).

Anther example:

Classification Issue:

The Iris set is used.

```
def load_iris_data():
    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target
    return cross_validation.train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def test_DTClassifier(*data):
    x_train, x_test, y_train, y_test = data
    clf = DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    print('train score:%f' % (clf.score(x_train, y_train)))
    print('test score: %f' %(clf.score(x_test, y_test)))

x_train, x_test, y_train, y_test = load_iris_data()
test_DTClassifier(x_train, x_test, y_train, y_test)

```

```
train score:1.000000
test score: 0.974359
```

It works well.

## CART

Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.

## TIPS

Sometimes continuous values and incomplete values bring troubles. As to he continuous values, bi-partition can be applied. Setting weight is used to deal with incomplete values.
