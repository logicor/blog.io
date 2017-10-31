---
layout: post
title: Machine Learning Note(6)
date: 2017-10-31
categories: blog
tags: [note]
description: Machine Learning
---

# Bayes Classifier

In statistical classification the Bayes classifier minimizes the probability of misclassification.

## Bayes Rules

In probability theory and statistics, Bayes’ theorem (alternatively Bayes’ law or Bayes' rule) describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if cancer is related to age, then, using Bayes’ theorem, a person’s age can be used to more accurately assess the probability that they have cancer, compared to the assessment of the probability of cancer made without knowledge of the person's age.

One of the many applications of Bayes' theorem is Bayesian inference, a particular approach to statistical inference. When applied, the probabilities involved in Bayes' theorem may have different probability interpretations. With the Bayesian probability interpretation the theorem expresses how a subjective degree of belief should rationally change to account for availability of related evidence. Bayesian inference is fundamental to Bayesian statistics.

Bayes’ theorem is named after Reverend Thomas Bayes (/beɪz/; 1701–1761), who first provided an equation that allows new evidence to update beliefs in his An Essay towards solving a Problem in the Doctrine of Chances (1763). It was further developed by Pierre-Simon Laplace, who first published the modern formulation in his 1812 "Théorie analytique des probabilités". Sir Harold Jeffreys put Bayes’ algorithm and Laplace's formulation on an axiomatic basis. Jeffreys wrote that Bayes' theorem "is to the theory of probability what the Pythagorean theorem is to geometry".[wiki]

According to different type of feather conditional probability, there are many kinds of classifiers.

Here we use the digit dataset

```
def show_digits():
    digits = datasets.load_digits()
    fig = plt.figure()
    print("vector from images 0", digits.data[0])
    for i in range(25):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow(digits.images[i], cmap='gray_r',interpolation='nearest')
    plt.show()
```

![result](http://oybqmhgid.bkt.clouddn.com/Figure_3_1.png)

## GaussianNB

Gaussian distribution 

```
def test_GaussianNB(*data):
    x_train, x_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()
    cls.fit(x_train, y_train)
    print('train score:%.2f' % (cls.score(x_train, y_train)))
    print('test score: %.2f' % (cls.score(x_test, y_test)))

x_train, x_test, y_train, y_test = load_data()
test_GaussianNB(x_train, x_test, y_train, y_test)
```
The result is :

```
train score:0.86
test score: 0.83
```

## MultinomialNB

Polynomial distribution 

```
def test_GaussianNB(*data):
    x_train, x_test, y_train, y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(x_train, y_train)
    print('train score:%.2f' % (cls.score(x_train, y_train)))
    print('test score: %.2f' % (cls.score(x_test, y_test)))

x_train, x_test, y_train, y_test = load_data()
test_GaussianNB(x_train, x_test, y_train, y_test)
```

Result:

```
train score:0.91
test score: 0.91
```

## BernouliNB

Binomial distribution

```
def test_GaussianNB(*data):
    x_train, x_test, y_train, y_test = data
    cls = naive_bayes.BernoulliNB()
    cls.fit(x_train, y_train)
    print('train score:%.2f' % (cls.score(x_train, y_train)))
    print('test score: %.2f' % (cls.score(x_test, y_test)))

x_train, x_test, y_train, y_test = load_data()
test_GaussianNB(x_train, x_test, y_train, y_test)
```

Result:

```
train score:0.87
test score: 0.85
```