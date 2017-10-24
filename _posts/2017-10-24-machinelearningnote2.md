---
layout: post
title: Machine Learning Note(2)
date: 2017-10-24
categories: blog
tags: [note]
description: Machine Learning
---

# Regularization of linear regression model

As to Multivariate Linear Regression, Regularization is introduced to minimize the mean square error. Regularization means priori hypotheses.There are three commonly used regularization methods



## Ridge Regression

<img src="http://www.forkosh.com/mathtex.cgi? \widehat{L} = L + \alpha \left \| \overrightarrow{w} \right \|_{2}^{2},\alpha \geq 0">

Function code:

`def test_Ridge(*data):
	x_train,x_test,y_train,y_test = data
	regr = linear_model.Ridge()
	regr.fit(x_train,y_train)
    print('coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    print('score : %.2f' % regr.score(x_test, y_test))
test_Ridge(x_train, x_test, y_train, y_test)`

The results are as follows:

`coefficients:[  21.19927911  -60.47711393  302.87575204  179.41206395    8.90911449
  -28.8080548  -149.30722541  112.67185758  250.53760873   99.57749017], intercept 152.45
Residual sum of squares: 3192.33`

The result is bad.
When *a* takes different values, the results are as follows:

Function code:

`def test_Ridge_alpha(*data):
    [x_train, x_test, y_train, y_test] = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(x_train, y_train)
        scores.append(regr.score(x_test, y_test))
    # pic
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    plt.show()`

Here shows the pic:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_1.png)

The model is simple, however, the value *a* can not make a big difference.