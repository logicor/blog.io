---
layout: post
title: Machine Learning Note(1)
date: 2017-10-23
categories: blog
tags: [note]
description: Machine Learning
---

#Linear Model

Generally speaking, the linear model can be expressed by the equation as follow.
<img src="http://www.forkosh.com/mathtex.cgi? f(\overrightarrow{x})=\overrightarrow{w}\cdot \overrightarrow{x}+b">
In the formula, *w* corresponds to the weight of each vector.


##Multivariate Linear Regression

Regression analysis is essentially a function estimation problem.
Given <img src="http://www.forkosh.com/mathtex.cgi? \overrightarrow{x_{i}}">,
the predicted value is
<img src="http://www.forkosh.com/mathtex.cgi? \overrightarrow{y_{i}}=\overrightarrow{w}\cdot \overrightarrow{x_{i}}+b">
Then, the loss function is:
<img src="http://www.forkosh.com/mathtex.cgi? L(f)=\sum_{N}^{i=1}(\overrightarrow{w}\cdot \overrightarrow{x_{i}}+b-y_{i})
">
This function needs to be minimized by using the method of Partial Least-Square Regression.
I can get :
<img src="http://www.forkosh.com/mathtex.cgi? \overrightarrow{\widetilde{w}}^{\ast }=argmin(\overrightarrow{y}-\overrightarrow{x}\overrightarrow{\widetilde{w}})^{T}(\overrightarrow{y}-\overrightarrow{x}\overrightarrow{\widetilde{w}})">
Then differentiate:
<img src="http://www.forkosh.com/mathtex.cgi? \frac{\partial E_{\overrightarrow{\widetilde{w}}}}{\partial \overrightarrow{\widetilde{w}}}=2\overrightarrow{x}^{T}(\overrightarrow{x}\cdot \overrightarrow{\widetilde{w}}-\overrightarrow{y})=\overrightarrow{0})">
Then *w* can be calculated
In order to valuate prediction performance, the performance score is as follow:
<img src="score=1-\frac{\sum (y_{i}-\widetilde{y}_{i})^{2}}{(y_{i}-\overline{y})^{2}}">


##Example:

(I used data set from the scikit-learn tool.)

codes:

`import numpy as np
from sklearn import datasets, linear_model,cross_validation
`
`def load_data():
	diabetes = datasets.load_diabetes()
	return cross_validation.train_test_split(datasets.data,diabetes.target,test_size=0.25,random_state=0)
`
`def test_LinearRegression(*data):
	x_train,x_test,y_train,y_test = data
	regr = linear_model.LinearRegression()
	regr.fit(x_train,y_train)
	print('coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
	print("Residual sum of squares: %.2f"%np.mean((regr.predict(x_test)-y_test)**2))
	print('score : %.2f'%regr.score(x_test,y_test))
`
`x_train,x_test,y_train,y_test=load_data()
test_LinearRegression(x_train,x_test,y_train,y_test)
`

The results are as follows:

`coefficients:[ -43.26774487 -208.67053951  593.39797213  302.89814903 -560.27689824
  261.47657106   -8.83343952  135.93715156  703.22658427   28.34844354], intercept 153.07
Residual sum of squares: 3180.20
score : 0.36`

The score is just 0.36, because it is a traditional algorithm. It is not a effective algorithm.

