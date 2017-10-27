---
layout: post
title: Machine Learning Note(3)
date: 2017-10-27
categories: blog
tags: [note]
description: Machine Learning
---

# Regularization of linear regression model

In addition to ridge regression, there are two commonly used regularization.


## Lasso Regression

This algorithm can control and shrink the coefficient to zero. It is a very popular method.
(This algorithm is also included in SKLEARN)

```
def test_Lasso(*data):
    [x_train, x_test, y_train, y_test] = data
    regr = linear_model.Lasso()
    regr.fit(x_train, y_train)
    print('coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    print('score : %.2f' % regr.score(x_test, y_test))

test_Lasso(x_train, x_test, y_train, y_test)
```

The result is :
```
coefficients:[   0.           -0.          442.67992538    0.            0.            0.

   -0.            0.          330.76014648    0.        ], intercept 152.52
Ran 0 tests in 0.000s

Residual sum of squares: 3583.42
OK
score : 0.28
```

We can see the influence of parameters on results:

![result](http://oybqmhgid.bkt.clouddn.com/Figure_2.png)

The performance is better than ridge regression.

## ElasticNet Regression

This algrithom is a combination of two algorithms to achieve better results. It has two parameters, so it works better, but needs more debugging.

Here is a simple demo.

```
def test_ElasticNet(*data):
    [x_train, x_test, y_train, y_test] = data
    regr = linear_model.ElasticNet()
    regr.fit(x_train, y_train)
    print('coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    print('score : %.2f' % regr.score(x_test, y_test))
test_ElasticNet(x_train, x_test, y_train, y_test)
```

The result is awful.

```
coefficients:[ 0.40560736  0.          3.76542456  2.38531508  0.58677945  0.22891647
 -2.15858149  2.33867566  3.49846121  1.98299707], intercept 151.93

Residual sum of squares: 4922.36

score : 0.01
```

Its score is only 0.01

Let's see the varity of results with the parameters

```
def test_ElasticNet_alpha_rho(*data):
    [x_train, x_test, y_train, y_test] = data
    alphas = np.logspace(-2, 2)
    rhos = np.linspace(0.01, 1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
            regr.fit(x_train, y_train)
            scores.append(regr.score(x_test, y_test))
    # pic
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap='jet',
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()

test_ElasticNet_alpha_rho(x_train, x_test, y_train, y_test)
```

The result likes a beautiful wave.

![result](http://oybqmhgid.bkt.clouddn.com/Figure_3.png)
