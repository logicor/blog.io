---
layout: post
title: Machine Learning Note(7)
date: 2017-11-01
categories: blog
tags: [note]
description: Machine Learning
---

# K-nearest neighbors algorithm

In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression.In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression。

k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.

Both for classification and regression, a useful technique can be to assign weight to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.[wiki]

Classifier:

```
def test_KNN(*data):
    x_train, x_test, y_train, y_test = data
    cls = neighbors.KNeighborsClassifier()
    cls.fit(x_train, y_train)
    print('train score:%f' % (cls.score(x_train, y_train)))
    print('test score: %f' % (cls.score(x_test, y_test)))

x_train, x_test, y_train, y_test = load_data()
test_KNN(x_train, x_test, y_train, y_test)
```

Result:

```
train score:0.87
test score: 0.85
```

Regreesor:

```
def test_KNN_reg(*data):
    x_train, x_test, y_train, y_test = data
    cls = neighbors.KNeighborsRegressor()
    cls.fit(x_train, y_train)
    print('train score:%f' % (cls.score(x_train, y_train)))
    print('test score: %f' % (cls.score(x_test, y_test)))


x_train, x_test, y_train, y_test = creat_reg_data(100)
test_KNN_reg(x_train, x_test, y_train, y_test)
```

Result:

```
train score:0.980171

Ran 0 tests in 0.000s
test score: 0.961496
```


## Three elements of KNN

K value selection:

The best choice of k depends upon the data; generally, larger values of k reduce the effect of noise on the classification, but make boundaries between classes less distinct. A good k can be selected by various heuristic techniques (see hyperparameter optimization). The special case where the class is predicted to be the class of the closest training sample (i.e. when k = 1) is called the nearest neighbor algorithm.

Classifier:

```
def test_knn_k(*data):
    x_train, x_test, y_train, y_test = data
    ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')
    weights = ['uniform', 'distance']

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for weight in weights:
        tr_scores = []
        te_scores = []
        for k in ks:
            cls = neighbors.KNeighborsClassifier(weights=weight,n_neighbors=k)
            cls.fit(x_train, y_train)
            te_scores.append(cls.score(x_test,y_test))
            tr_scores.append(cls.score(x_train,y_train))
        ax.plot(ks,te_scores,label="testing_scores:weight=%s"%weight)
        ax.plot(ks,tr_scores,label="training_scores:weight=%s"%weight)
    ax.legend(loc='best')
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNNclassifier")
    plt.show()

x_train, x_test, y_train, y_test = load_data()
test_knn_k(x_train, x_test, y_train, y_test)
```

Result:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_5_1.png)

With the increase of k, the performance becomes poorer and poorer. And 'uniform' mode means weights of all the samples are same, the result will be disturbed when the k is bigger and bigger. The 'distance' mode means the weights are related to the position of the samples. This mode can balance the value k and the distance.

Regreesor:

```
def test_knnreg_k(*data):
    x_train, x_test, y_train, y_test = data
    ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')
    weights = ['uniform', 'distance']

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for weight in weights:
        tr_scores = []
        te_scores = []
        for k in ks:
            cls = neighbors.KNeighborsRegressor(weights=weight,n_neighbors=k)
            cls.fit(x_train, y_train)
            te_scores.append(cls.score(x_test,y_test))
            tr_scores.append(cls.score(x_train,y_train))
        ax.plot(ks,te_scores,label="testing_scores:weight=%s"%weight)
        ax.plot(ks,tr_scores,label="training_scores:weight=%s"%weight)
    ax.legend(loc='best')
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNNRegressor")
    plt.show()
x_train, x_test, y_train, y_test = creat_reg_data(100)
test_knnreg_k(x_train, x_test, y_train, y_test)
```

Result:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_5_2.png)

The result is same as the classifer.

Distance calculation:

Distance is a reflection of similarity.

Classifier:

```
def test_knn_p(*data):
    x_train, x_test, y_train, y_test = data
    ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')
    ps = [1,2,10]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for p in ps:
        tr_scores = []
        te_scores = []
        for k in ks:
            cls = neighbors.KNeighborsClassifier(p=p,n_neighbors=k)
            cls.fit(x_train, y_train)
            te_scores.append(cls.score(x_test,y_test))
            tr_scores.append(cls.score(x_train,y_train))
        ax.plot(ks,te_scores,label="testing_scores:p=%s"%p)
        ax.plot(ks,tr_scores,label="training_scores:p=%s"%p)
    ax.legend(loc='best')
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNNclassifier")
    plt.show()

x_train, x_test, y_train, y_test = load_data()
test_knn_p(x_train, x_test, y_train, y_test)
```

Result:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_5_3.png)

Regressor:

```
def test_knnreg_p(*data):
    x_train, x_test, y_train, y_test = data
    ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')
    ps = [1,2,10]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for p in ps:
        tr_scores = []
        te_scores = []
        for k in ks:
            cls = neighbors.KNeighborsRegressor(p=p,n_neighbors=k)
            cls.fit(x_train, y_train)
            te_scores.append(cls.score(x_test,y_test))
            tr_scores.append(cls.score(x_train,y_train))
        ax.plot(ks,te_scores,label="testing_scores:p=%s"%p)
        ax.plot(ks,tr_scores,label="training_scores:p=%s"%p)
    ax.legend(loc='best')
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNNRegressor")
    plt.show()
x_train, x_test, y_train, y_test = creat_reg_data(100)
test_knnreg_p(x_train, x_test, y_train, y_test)
```

Result:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_5_4.png)

We can see that the ways to calculate distance make no different.

Classification rule :

Majority voting is commonly applied. Sometimes weight is considered.

## K-D tree

The k-d tree is a binary tree in which every node is a k-dimensional point. Every non-leaf node can be thought of as implicitly generating a splitting hyperplane that divides the space into two parts, known as half-spaces. Points to the left of this hyperplane are represented by the left subtree of that node and points right of the hyperplane are represented by the right subtree. The hyperplane direction is chosen in the following way: every node in the tree is associated with one of the k-dimensions, with the hyperplane perpendicular to that dimension's axis. So, for example, if for a particular split the "x" axis is chosen, all points in the subtree with a smaller "x" value than the node will appear in the left subtree and all points with larger "x" value will be in the right subtree. In such a case, the hyperplane would be set by the x-value of the point, and its normal would be the unit x-axis.