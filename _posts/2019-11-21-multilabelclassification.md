---
title: "Math behind multilabel classification"
date: 2019-11-20
tags: [machine learning, data science]
mathjax: "true"
---
###Introduction

We often come across classification problem in which our response variable takes more than two possible discrete outcomes. For example rather than classifying our mails into two classes spam or not spam we might want to classify it into three classes such as spam, personal mail and work-related mail.
Let's try to formalize this by looking into the probabilistic interpretation of this machine learning problem where we will assume the probability distribution of our response variable as multinomial.

In this classification settings in the training set \\( {(x^{1},y^{1}),(x^{2},y^{2}),....,(x^{m},y^{m})} \\) response variable y can take K different values so \\( y^{m} \in {1,2...K} \\) and we want our model to estimate the probability that \\( P(y=K\mid x) \\) for each value of \\( k=1,2,...K \\) i.e we want to estimate the probability of class label taking on each of the \\(K \\) different values conditioned on input \\( x \\). So our model hypothesis function takes the form below :

$$
h\theta_x=\begin{bmatrix} P(y=1|x;\theta)
\\ P(y=2|x;\theta)
\\ P(y=3|x;\theta)
\\.
\\.
\\P(y=K|x;\theta)
\end{bmatrix}
$$


We will first parameterize the probability of our response variable taking on \\( K \\) possible outcomes. For this we can use \\( K \\) parameters \\( \phi_1,\phi_2.....\phi_k \\) specifying the probability of each of the outcomes. But if we use all the parameters it will be redundant as knowing \\( K \\) will uniquely determine last one as they must satisfy \\( \sum_{1}^{k}\phi_i=1 \\). So instead we will parametrize the response variable with only \\( k-1 \\) parameters \\( \phi_i....\phi_k-1 \\).

So our model assumes that conditional distribution of y given x is given by :

$$
P(y=i\mid x;\theta)=\phi_i
$$

$$
\phi_i=\frac{e^{\theta_i^T}x}{\sum_{j=1}^{k}e^{\theta_j^T}x}
$$

To obtain the second line above you can go through reference shared below, this relationship is also known as SoftMax function. Here \\( \theta_i \\) are the parameters of our model but notice that \\( \theta_i \\) is a vector quantity since for K classes we will have to fit \\( \theta_1,\theta_2.....\theta_k \\) parameters. We can write our model hypothesis function and it's expected output as :

$$
h_\theta_x=\begin{bmatrix} \phi_1
\\ \phi_2
\\ \phi_3
\\.
\\.
\\\phi_k-1
\end{bmatrix} = \begin{bmatrix} \frac{e^{\theta_1^T}x}{\sum_{j=1}^{k}e^{\theta_j^T}x}
\\ \frac{e^{\theta_2^T}x}{\sum_{j=1}^{k}e^{\theta_j^T}x}
\\ \frac{e^{\theta_3^T}x}{\sum_{j=1}^{k}e^{\theta_j^T}x}
\\.
\\.
\\ \frac{e^{\theta_{k-1}^T}x}{\sum_{j=1}^{k}e^{\theta_j^T}x}
\end{bmatrix}
$$

### Parameter fitting

We now describe the parameter fitting but before that let's try to understand indicator function, indicator function \\( \1{.} \\) takes on value 1 if it's argument is true otherwise 0. For example \\( 1{2+2=4} = 1  \\) and \\( 1{3=2} = 0 \\).

For fitting the parameters of the model we can use maximum likelihood estimates of \\( \theta_i \\) which are obtained by finding \\( \theta_i \\) which maximizes the likelihood of our data. We can write the Likelihood \\( l(\theta) \\) of the data :

$$
l(\theta)=\prod_i^{m} P(y^{i} \mid x;\theta)
$$

The above expression comes from i.i.d assumption on the response variable, now if we take the log at both side in the above equation :

$$
 \log l(\theta) = \sum_i^{m} \log P(y^{i} \mid x;\theta)
 =\sum_i^m \log\prod_l^{k} {\frac{e^{\theta_l^T}x^{i}}{\sum_{j=1}^{k}e^{\theta_j^T}x^{i}}}^1{y^{i}=l}
 $$

 Now to find the parameters \\( \theta \\) that maximizes the \\( \log \\) likelihood we can use iterative optimization algorithm like gradient descent or newton's method. To see the implementation in python check out my jupyter notebook.
