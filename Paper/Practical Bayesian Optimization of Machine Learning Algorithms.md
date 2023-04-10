# **Practical Bayesian Optimization of Machine Learning Algorithms**






- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. In Advances in Neural Information Processing Systems (pp. 2951-2959). (https://arxiv.org/abs/1206.2944)






<br>

<br>

# 0. Abstract

- **Machine learning algorithms** frequently require careful tuning of
    - model hyperparameters, 
    - regularization terms, 
    - optimization parameters.


<br>


- This tuning is often a ◼️ “black art” ◼️

- Much more appealing is the idea of developing automatic approaches which can optimize the performance of a given learning algorithm to the task at hand.

- 좋은 아이디어는 직접 손으로 하는 것보다 이것을 자동화 하는 것이다.

<br>

- ✨ In this work, we consider the automatic tuning problem within the framework of Bayesian optimization. ✨ 
    - It is a learning algorithm’s generalization performance is modeled as a sample from a Gaussian process (GP). 



- Here we show how the effects of the Gaussian process prior and the associated inference procedure can have a <u>large impact</u> on the success or failure of Bayesian optimization.







- We show that these proposed algorithms improve on previous automatic procedures and can "reach or surpass" human expert-level optimization on a diverse set of contemporary algorithms including latent Dirichlet allocation, structured SVMs and convolutional neural networks.

- Prior로 설정한 GP는 큰 역할을 한다.

<br>

<br>

# 1. Introduction

- 머신러닝 알고리즘은 파라미터가 없기가 힘들다

    - The properties of a regularizer
    - the hyperior of a generative model
    - the step size of a gradient-based optimization

- Learning procedures almost always require a set of high-level choices that significantly impact generalization performance.

- 이러한 조절장치 최소한으로 하는게 좋다.

<br>

- 고수준 파라미터의 최적화 문제를 자동화하는 더 유연한 방법은, 이러한 튜닝을 일반화 성능을 반영하는 알 수 없는 <u>블랙박스 함수의 최적화로 간주</u>하고, 이를 위해 개발된 알고리즘을 사용하는 것.

- 이러한 최적화 문제는 학습 절차의 저수준 목적 함수와는 다른 특징을 가지며, 여기에서 함수 평가는 기본 기계 학습 알고리즘을 완료하는 것을 필요로 하므로 매우 비용이 많이 든다. 

    - 💡 **베이지안 최적화** 💡

    - 베이지안 최적화는 알 수 없는 함수가 가우시안 프로세스(GP)에서 샘플링된 것으로 가정하고, 하이퍼파라미터 값을 조정하면서 일반화 성능을 측정.

<br>

- 하지만! 
- 기계 학습 알고리즘은 다른 최적화 문제와 달리 각 함수 평가에 걸리는 시간이 다르고, 비용 개념을 최적화 절차에 포함시키는 것이 바람직하다
- 병렬 처리를 이용해 보다 빠르게 최적의 해결책에 도달할 수 있는 베이지안 최적화 절차를 개발



<br>







- 이 논문의 기여는 두 가지 
    1. 기계 학습 알고리즘의 베이지안 최적화를 위한 좋은 방법론을 도출
        - 기계 학습 알고리즘에 대한 베이지안 최적화 방법론을 제안하며, 커널 매개변수의 완전한 베이지안 처리가 결과의 탄력성에 대한 중요성을 강조
        - in contrast to the more standard procedure of optimizing hyperparameters (e.g. Bergstra et al. (2011)).

    2. 비용(cost) 개념을 실험에 반영하는 새로운 알고리즘을 제시
        - 비용 개념을 고려한 새로운 알고리즘을 제안하고, 병렬 처리를 이용해 더 빠른 최적의 결과를 얻을 수 있는 알고리즘도 제시


<br>

<br>


# 2. Bayesian Optimization with Gaussian Process Priors.





- We are interested in finding the minimum of a function $f(\mathbf{x})$ on some bounded set $\mathcal{X}$, which we will take to be a subset of $\mathbb{R}^D.$

- 다른 Optimization과 다른점: 
    - It constructs a probabilistic model for $f(\mathbf x)$ and then exploits this model to make decisions about where in $\mathcal{X}$ to evaluate the function.
    - ➕ Uncertainty




- **The essential philosophy**: 
    - to use all of the information available from previous evaluations of $f(\mathbf{x})$ 
    - and <u> not simply rely on local gradient and Hessian approximations.</u>

<br>


- 만약 $f(\mathbf{x})$가 평가하기에 시간이 많이 든다고 했을 때 (머신러닝 알고리즘 같은 경우)도 좋다.


- 먼저 General Bayesian optimization approach를 간단히 리뷰하자.














<br>

## 2.1 Gaussian Processes

- The **Gaussian process (GP)** is a convenient and powerful prior
distribution on functions

- which we will take here to be of the form $$f: \mathcal{X} \rightarrow \mathbb{R}.$$

- The GP is defined by the property that any finite set of $N$ points $$\{ \mathbf{x}\in \mathcal{X} \}^N_{n=1}$$ induces a multivariate Gaussian distribution on $\mathbb{R}^N$

- The $n$ th of these points is taken to be the function value $f(\mathbf{x}_n)$

- 가우시안 분포의 특징 중 하나인 마진화(marginalization properties)를 이용하면 조건부 및 주변 확률을 간단하게 계산할 수 있다    
 
- The support and properties of the resulting dis tribution on functions are determined by 
    - a mean function $m: \mathcal{X} \rightarrow \mathbb{R}$ and
    - a positive definite covariance function $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ 


## 2.2. Acquisition Functions for Bayesian Optimization

> "Acquisition Functions for Bayesian Optimization"은 베이지안 최적화에서 사용되는 효율적인 실험 실행 방법 중 하나인 획득 함수(acquisition function)에 대한 논문입니다. <u>베이지안 최적화에서 획득 함수는 현재까지 수집한 데이터로부터 새로운 데이터를 수집할 위치를 결정합니다.</u> 이 논문에서는 여러 가지 획득 함수가 소개되며, 효율적인 최적화 알고리즘의 구축을 위한 다양한 기술과 방법이 제안됩니다. 획득 함수는 베이지안 최적화의 성능을 결정하는 중요한 요소 중 하나이므로, 이 논문은 베이지안 최적화를 사용하는 많은 연구자와 엔지니어들에게 유용한 정보를 제공합니다.

- We assume that the function $f(\mathbf{x})$ is drawn from a GP prior,
- and that our observations are of the form $\{ \mathbf{x}_n, y_n \}^N_{n=1}$, where $y_n \sim \mathcal{N}(f(\mathbf{x}_n, \nu))$
    - $\nu$ is 






