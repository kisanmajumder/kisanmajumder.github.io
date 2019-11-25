---
title: "Understanding the math behind multilabel classification"
date: 2019-11-20
tags: [machine learning, data science]
mathjax: "true"
---

We often come across classification problem in which our response variable takes more than two possible discrete outcomes. For example rather than classifying our mails into two classes spam or not spam we might want to classify it into three classes such as spam, personal mail and work-related mail.
In this classification settings our training set \\( {(x^{1},y^{1}),(x^{2},y^{2}),....,(x^{m},y^{m})} \\) and response variable y can take K different values so \\( y^{m} \in {1,2...K} \\).

We want our model hypothesis function to estimate the probability that P(y=K|x) for each value of k=1,2,3....K, i.e we want to estimate the probability of class label taking on each of the K different values. Thus our hypothesis \\( h_\theta(x) \\) takes the form :

\\[ \h_\theta(x)=\begin{bmatrix} P(y=1|x;\theta)
\\ P(y=2|x;\theta)
\\ P(y=3|x;\theta)
\\.
\\.
\\P(y=K|x;\theta)
\end{bmatrix} \\]
