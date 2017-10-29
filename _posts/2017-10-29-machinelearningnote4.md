---
layout: post
title: Machine Learning Note(4)
date: 2017-10-27
categories: blog
tags: [note]
description: Machine Learning
---

# Regularization of linear regression model

 This model can also be used for classification problems. In order to solve two classification problem, for example, conditional probability can be predicted with the linear model.

## Logistic Regression 

The generalized linear model is adopted. The best choice is unit step function：


<img src="http://www.forkosh.com/mathtex.cgi? P(y=1/\overrightarrow{x})=\left\{\begin{matrix}0, &z<0\\0.5, &z=0\\1, &z>0\end{matrix}\right">

But the function is non-derivable, we use logisitic function instead of that.

<img src="http://www.forkosh.com/mathtex.cgi? p(y=1/\overrightarrow{x})=\frac{1}{1-e^{-z}},z=\overrightarrow{w}\cdot \overrightarrow{x}+b">

Here:

<img src="http://www.forkosh.com/mathtex.cgi? p(y=1/\overrightarrow{\widetilde{x}})=\pi (\overrightarrow{\widetilde{x}})">

Likelihood function is :

<img src="http://www.forkosh.com/mathtex.cgi? L(\overrightarrow{\widetilde{w}})=\sum_{i=1}^{N}[y_{i}log\frac{\pi(\overrightarrow{\widetilde{x_{i}}})}{1-\pi(\overrightarrow{\widetilde{x_{i}}})}+log(1-\pi(\overrightarrow{\widetilde{x_{i}}}))]">


 The parameters are estimated by seeking extreme values of the function.

 Example:

```
def load_iris_data():
    iris = datasets.load_iris()
    x_train = iris.data
    y_train = iris.target
    return cross_validation.train_test_split(x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)

def test_LogisticRegression(*data):
    [x_train, x_test, y_train, y_test] = data
    regr = linear_model.LogisticRegression()
    regr.fit(x_train, y_train)
    print('coefficients:%s, intercept %.s' % (regr.coef_, regr.intercept_))
    print('score : %.2f' % regr.score(x_test, y_test))

data = load_iris_data()
x_train, x_test, y_train, y_test = data
```

The result is good：

```
coefficients:[[ 0.40769719  1.32793253 -2.12687162 -0.96614355]
 [ 0.1932691  -1.31070419  0.60821724 -1.19814744]
 [-1.50100362 -1.33529511  2.16377642  2.23963779]], intercept 
score : 0.97
```

 The proportion of misclassification is 3% 
 The default setting is one-vs-rest, we change the setting with multi-class:

```
def test_LogisticRegression_multinomial(*data):
    [x_train, x_test, y_train, y_test] = data
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(x_train, y_train)
    print('coefficients:%s, intercept %.s' % (regr.coef_, regr.intercept_))
    print('score : %.2f' % regr.score(x_test, y_test))
data = load_iris_data()
x_train, x_test, y_train, y_test = data
```

We have got a perfect result:

```
coefficients:[[-0.36834503  0.84161824 -2.2786534  -0.98934481]
 [ 0.34136173 -0.33359848 -0.031646   -0.82947433]
 [ 0.02698329 -0.50801976  2.3102994   1.81881915]], intercept 
score : 1.00
```

Here we try to figure out the influence of regularization coefficient weights.

The result is as follow:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_4.png)
 

## Linear Discriminant Analysis (LDA)

Linear discriminant analysis (LDA) is a generalization of Fisher's linear discriminant, a method used in statistics, pattern recognition and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.[Wiki]

In order to minimize the distance between similar classes and maximize the distance betweenm different classes, the main idao of LDA can be explain as：

<img src="http://www.forkosh.com/mathtex.cgi? J=\frac{\left\|\overrightarrow{w}^{T}\overrightarrow{\mu_{0}}-\overrightarrow{w}^{T}\overrightarrow{\mu_{1}}\right\|_{2}^{2}}{\overrightarrow{w}^{T}\sum _{0}\overrightarrow{w}+\overrightarrow{w}^{T}\sum _{1}\overrightarrow{w}}">

Define within-class and between-class scatter matrix:

<img src="http://www.forkosh.com/mathtex.cgi? J=\frac{\overrightarrow{w}^{T}S_{b}\overrightarrow{w}}{\overrightarrow{w}^{T}S_{w}\overrightarrow{w}}">

Obtaining the optimized result as modifying and optimizing J, it is with simple, quick and convenient advantages.

```
def test_LDA(*data):
    [x_train, x_test, y_train, y_test] = data
    regr = discriminant_analysis.LinearDiscriminantAnalysis()
    regr.fit(x_train, y_train)
    print('coefficients:%s, intercept %.s' % (regr.coef_, regr.intercept_))
    print('score : %.2f' % regr.score(x_test, y_test))

data = load_iris_data()
x_train, x_test, y_train, y_test = data
test_LDA(x_train, x_test, y_train, y_test)
``` 

The result is good:

```
coefficients:[[  7.08232947   9.34752865 -14.9939558  -20.80332605]
 [ -2.17411651  -3.40669222   4.461538     2.72022083]
 [ -4.90821296  -5.94083643  10.5324178   18.08310522]], intercept 
score : 1.00
```

Try to figure the data after dimension reduction:

```
def plot_LDA(converted_x,y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers='o*s'
    for target, color, marker in zip([0,1,2],colors, markers):
        pos = (y == target).ravel()
        X = converted_x[pos,:]
        ax.scatter(X[:,0], X[:,1],X[:,2],color=color,marker=marker,label="Label %d"%target)
    ax.legend(loc="best")
    fig.suptitle("Iris after LDA")
    plt.show()
data = load_iris_data()
x_train, x_test, y_train, y_test = data
X=np.vstack((x_train,x_test))
Y=np.vstack((y_train.reshape(y_train.size,1),y_test.reshape(y_test.size,1)))
lda=discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X,Y)
converted_x = np.dot(X,np.transpose(lda.coef_))+lda.intercept_
plot_LDA(converted_x,Y)

```

![result](http://oybqmhgid.bkt.clouddn.com/Figure_5.png)

The picture shows the perfect performance of LDA.

Here we try to introduce regularization term.

```
def test_LDA_shrinkage(*data):
    [x_train, x_test, y_train, y_test] = data
    shrinkages = np.linspace(0,1.0,num=20)
    scores = []
    for shrinkage in shrinkages:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',shrinkage=shrinkage)
        lda.fit(x_train,y_train)
        scores.append(lda.score(x_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(shrinkages, scores)
    ax.set_xlabel(r"$\shrinkage")
    ax.set_ylabel(r"score")
    ax.set_title("LDA")
    ax.set_ylim(0,1.05)
    plt.show()

```

 The accuracy decreases with the increase of regularization term.

![result](http://oybqmhgid.bkt.clouddn.com/Figure_6.png)

