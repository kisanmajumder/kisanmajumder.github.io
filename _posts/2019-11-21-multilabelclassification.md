---
title: "Understanding the math behind multilabel classification"
date: 2019-11-20
tags: [machine learning, data science]
mathjax: "true"
---

We often come across classification problem in which our response variable takes more than two possible discrete outcomes. For example rather than classifying our mails into two classes spam or not spam we might want to classify it into three classes such as spam, personal mail and work-related mail.
Let's try to formalize this by looking into the probabilistic interpretation of the model where we will assume the distribution of our response variable as multinomial.

In this classification settings in our training set \\( {(x^{1},y^{1}),(x^{2},y^{2}),....,(x^{m},y^{m})} \\) the response variable y can take K different values so \\( y^{m} \in {1,2...K} \\).We want our model hypothesis function to estimate the probability that \\( P(y=K\mid x) \\) for each value of k=1,2,3....K, i.e we want to estimate the probability of class label taking on each of the *K* different values conditioned on input \\( x \\) Thus our hypothesis \\( h_\theta(x) \\) takes the form :

$$
h_\theta(x)=\begin{bmatrix} P(y=1|x;\theta)
\\ P(y=2|x;\theta)
\\ P(y=3|x;\theta)
\\.
\\.
\\P(y=K|x;\theta)
\end{bmatrix}
$$

To parameterize a multinomial over *K* possible outcomes we could use *K* parameters \\( \phi_1,\phi_2.....\phi_k \\) specifying the probability of each of the outcomes. But if we use all the parameters it will be redundant as knowing *k-1* will uniquely determine last one as they must satisfy (\\ \sum_{1}^{k}\phi_i = 1 \\). So instead we will parametrize the multinomial with only *k-1* parameters \\( \phi_i....\phi_k-1 \\).

Our model assumes that conditional distribution of y given x is given by :
$$
p(y=i\mid x;\theta)=\phi_i
                  =\frac{e^{\theta_i^T}x}{\sum_{j=1}{k}e^{\theta_j^T}x}

$$
