# **Practical Bayesian Optimization of Machine Learning Algorithms**






- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. In Advances in Neural Information Processing Systems (pp. 2951-2959). (https://arxiv.org/abs/1206.2944)


- *베이지안 최적화(Bayesian optimization)는 값들을 관측해가며 최적의 함수를 찾아가는 글로벌 최적화 기법입니다. 주요 목표는 가능한 한 적은 수의 관측을 통해 목표 함수를 최적화하는 것입니다. 이 방법은 블랙박스 함수에 대해 효과적으로 작동하며, 이러한 함수는 계산 비용이 많이 들거나, 그래디언트 정보가 없거나, 노이즈가 있는 경우에 사용됩니다.*

<details>
<summary> <span style="color: blue; text-decoration: underline;"> 이거랑 hyper-parameter를 찾는거랑은 무슨 연관이 있지? 어떻게 hyper-parameter를 찾는거야? </span> </summary>

- 베이지안 최적화는 하이퍼파라미터 최적화에 널리 사용되는 방법 중 하나입니다. 머신러닝 모델의 성능은 선택된 하이퍼파라미터에 크게 영향을 받습니다. 하이퍼파라미터는 모델 학습 과정에서 자동으로 업데이트되지 않는 모델의 설정 값입니다. 예를 들어, 신경망의 층(layer) 수, 뉴런의 수, 학습률(learning rate), 정규화(regularization) 계수 등이 있습니다.

- 하이퍼파라미터 최적화의 목표는 이러한 하이퍼파라미터를 조정하여 모델의 성능을 최대한 높이는 것입니다. 베이지안 최적화를 사용하여 하이퍼파라미터를 찾는 과정은 다음과 같습니다.

- 목표 함수 정의: 목표 함수는 모델의 성능을 측정하는 지표를 사용해 하이퍼파라미터 설정에 대한 점수를 계산합니다. 예를 들어, 교차 검증(cross-validation)을 사용하여 주어진 하이퍼파라미터 설정에 대한 모델 성능을 평가할 수 있습니다.

- 서로게이트 모델 학습: 베이지안 최적화를 사용하여 목표 함수의 불확실성을 모델링하고, 가우시안 프로세스와 같은 서로게이트 모델을 사용하여 하이퍼파라미터 공간에서 최적의 지점을 찾습니다.

- Acquisition Function 사용: 탐색-활용 트레이드오프를 고려하여 다음 하이퍼파라미터 설정을 선택합니다. 예를 들어, Probability of Improvement (PI), Expected Improvement (EI), 또는 Upper Confidence Bound (UCB)를 사용할 수 있습니다.

- 새로운 하이퍼파라미터 설정 평가: 선택한 하이퍼파라미터 설정을 사용하여 모델을 학습시키고 목표 함수를 평가합니다.

- 반복: 새로운 하이퍼파라미터 설정과 목표 함수 값을 데이터에 추가하고, 서로게이트 모델을 업데이트한 다음, 다음 하이퍼파라미터 설정을 선택합니다. 이 과정은 미리 정의된 종료 조건이 충족될 때까지 반복됩니다(예: 반복 횟수, 허용 오차 등).

- 이 과정을 통해 베이지안 최적화는 하이퍼파라미터 공간에서 성능을 최적화하는 설정을 찾아갑니다. 최종적으로 베이지안 최적화가 완료되면, 가장 높은 목표 함수 값을 얻은 하이퍼파라미터 설정을 선택합니다. 이렇게 선택된 하이퍼파라미터를 사용하여 모델을 학습시키면, 일반적으로 검증 데이터셋에 대해 더 나은 성능을 보이게 됩니다.

- 베이지안 최적화는 하이퍼파라미터 최적화에 사용되는 다양한 방법 중 하나로, 그리드 탐색(grid search)이나 랜덤 탐색(random search)에 비해 목표 함수 평가 횟수를 줄이는 데 효과적입니다. 이를 통해 계산 비용이 높은 머신러닝 모델의 하이퍼파라미터 최적화에 효율적으로 사용할 수 있습니다. 그러나 베이지안 최적화의 성능은 서로게이트 모델과 Acquisition Function의 선택에 따라 다소 달라질 수 있습니다. 따라서, 특정 문제에 대해 적합한 설정을 찾는 것이 중요합니다.
</details>

<details>
<summary> <span style="color: blue; text-decoration: underline;"> 3번 "Acquisition Function 사용" 단계에서 하이퍼파라미터 설정을 선택하는거야?</span> </summary>

네, 맞습니다. 3번 "Acquisition Function 사용" 단계에서는 하이퍼파라미터 설정을 선택하는 것이 핵심입니다. Acquisition Function은 서로게이트 모델(가우시안 프로세스 등)의 예측 결과와 불확실성 정보를 바탕으로, 탐색(Exploration)과 활용(Exploitation) 사이의 균형을 맞추어 하이퍼파라미터 설정을 선택합니다.

여기서 탐색(Exploration)은 하이퍼파라미터 공간에서 아직 평가되지 않은 영역을 샘플링하여 목표 함수의 불확실성을 줄이는 것을 의미합니다. 반면, 활용(Exploitation)은 현재까지 알려진 가장 좋은 하이퍼파라미터 설정 근처에서 샘플링하여 목표 함수의 값을 더 높이려는 것을 의미합니다.

Acquisition Function의 예로는 Probability of Improvement (PI), Expected Improvement (EI), 그리고 Upper Confidence Bound (UCB) 등이 있습니다. 이러한 함수들은 서로게이트 모델의 예측 평균과 분산을 입력으로 받아, 각 하이퍼파라미터 설정에 대한 점수를 계산합니다. 점수가 가장 높은 하이퍼파라미터 설정을 선택하여 다음 단계로 진행합니다.

따라서, Acquisition Function은 베이지안 최적화 과정에서 새로운 하이퍼파라미터 설정을 선택하는 역할을 합니다. 이를 통해 더 나은 성능을 얻을 가능성이 있는 하이퍼파라미터 설정을 찾아내는 것이 목표입니다.

</details>




<details>
<summary> <span style="color: blue; text-decoration: underline;"> 그럼 핵심은 함수값을 GP로 높고, 다음으로 관측할 지점을 어떤 decision rule을 통해 선택 한다음, 하이퍼 파라미터를 고르고, 실제 머신러닝 모델을 돌려보며 성능을 측정하는거야?</span> </summary>

네, 맞습니다. 베이지안 최적화의 핵심은 목표 함수의 값을 높이기 위해 하이퍼파라미터를 조정하는 것입니다. 이를 위해 아래와 같은 과정을 거칩니다.

서로게이트 모델(GP) 학습: 목표 함수의 값을 추정하기 위해 가우시안 프로세스(GP)와 같은 서로게이트 모델을 사용합니다. 이 모델은 목표 함수의 예측 평균과 불확실성(분산)을 제공합니다.

Acquisition Function 사용: 탐색-활용 트레이드오프를 고려하여 다음으로 관측할 지점을 선택합니다. 이를 위해 Probability of Improvement (PI), Expected Improvement (EI), Upper Confidence Bound (UCB) 등의 decision rule을 사용할 수 있습니다.

하이퍼파라미터 선택: Acquisition Function을 통해 계산된 점수를 기반으로 하이퍼파라미터 설정을 선택합니다.

실제 머신러닝 모델 평가: 선택된 하이퍼파라미터로 머신러닝 모델을 학습시키고, 성능을 측정합니다. 이를 통해 목표 함수의 값을 얻습니다.

반복: 새로운 하이퍼파라미터 설정과 목표 함수 값을 데이터에 추가하고, 서로게이트 모델을 업데이트한 다음, 다음 하이퍼파라미터 설정을 선택합니다. 이 과정은 미리 정의된 종료 조건이 충족될 때까지 반복됩니다.

이렇게 베이지안 최적화를 통해 최적의 하이퍼파라미터를 찾아 머신러닝 모델의 성능을 높이는 것이 목표입니다. 이 방법은 그리드 탐색이나 랜덤 탐색보다 더 효율적으로 하이퍼파라미터를 최적화할 수 있습니다.

</details>




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


## 0.1. Point 💡

- **Bayesian optimization**은 $f(\mathbf{x})$가 expensive black-box function일 때, 즉 한 번 input을 넣어서 output을 확인하는 것 자체가 cost가 많이 드는 function일 때 많이 사용하는 optimization method이다.

    - ✨✨✨ $\mathbf{x}$를 하이퍼 파라미터들
    - $f(\mathbf{x})$를 하이퍼 파라미터 $\mathbf{x}$를 사용했을 때의 모델의 성능 (ex. 정확도, error) 라고 생각하면 된다.✨✨✨

- **Bayesian optimization은 다음과 같은 방식으로 작동**

    1. ☝🏻먼저 지금까지 관측된 데이터들 $$D = \{(\mathbf{x}_1, f(\mathbf{x}_1)), (\mathbf{x}_2, f(\mathbf{x}_2)), \cdots \}$$ 를 통해, 전체 function $f(\mathbf{x})$를 어떤 방식을 사용해 estimate한다.

    2. ✌🏻 Function $f(\mathbf{x})$ 를 더 정밀하게 예측하기 위해 다음으로 관측할 지점 $$(\mathbf{x}_{n+1}, f(\mathbf{x}_{n+1}))$$ 을 어떤 decision rule을 통해 선택한다.

    3. 🤟🏻 새로 관측한 $(\mathbf{x}_{n+1}, f(\mathbf{x}_{n+1}))$ 을 $D$에 추가하고, 적절한 stopping criteria에 도달할 때 까지 다시 1로 돌아가 반복한다.


<br>

- 1에서 언급한 estimation을 할 때에는 $f(\mathbf{x})$가 Gaussian process prior를 가진다고 가정한 다음, posterior를 계산하여 function을 estimate한다.

- 2에서는 acquisition function $a( \mathbf{x} | D)$를 디자인해서 $\arg\max_{\mathbf{x}} a( \mathbf{x} | D)$ 를 계산해 다음 지점을 고른다.

<br>



- **Acquisition Function**


- Function $f(\mathbf{x})$가 GP prior를 가지는 Bayesian optimization을 진행 중이라고 가정해보자.

- $f(\mathbf{x})$의 모든 point x에 대해, 우리는 mean과 variance를 계산할 수 있다

- 이때 다음으로 관측해야할 부분이 어디인지 어떻게 알 수 있을까?

- 한 가지 방법은 estimated mean의 값이 가장 작은 지점은 관측하여 현재까지 관측된 값들을 기준으로 가장 좋은 점을 찾아보는 것

- 또 다른 방법은 variance의 값이 가장 큰 지점을 관측하여, 함수의 모양을 더 정교하게 탐색하는 방법

- 즉, 다음에 어떤 점을 탐색하느냐를 결정하는 문제는 explore-exploit 문제가 된다.
    - explore는 high variance point를 관측하는 것,
    - exploit은 low mean point를 관측하는 것

- Acquisition function이란 explore와 exploit을 적절하게 균형을 잡아주는 역할
    - Probability of Improvement
    - Expected Improvement
    - UCB
    









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

> "Acquisition Functions for Bayesian Optimization"은 베이지안 최적화에서 사용되는 효율적인 실험 실행 방법 중 하나인 획득 함수(acquisition function)에 대한 논문입니다. <u>베이지안 최적화에서 획득 함수는 현재까지 수집한 데이터로부터 새로운 데이터를 수집할 위치를 결정합니다.</u> ✨즉 다음에 시험해볼 하이퍼 파라미터를 결정✨ 이 논문에서는 여러 가지 획득 함수가 소개되며, 효율적인 최적화 알고리즘의 구축을 위한 다양한 기술과 방법이 제안됩니다. 획득 함수는 베이지안 최적화의 성능을 결정하는 중요한 요소 중 하나이므로, 이 논문은 베이지안 최적화를 사용하는 많은 연구자와 엔지니어들에게 유용한 정보를 제공합니다.

- We assume that the function $f(\mathbf{x})$ is drawn from a GP prior,
- and that our observations are of the form $\{ \mathbf{x}_n, y_n \}^N_{n=1}$, where $y_n \sim \mathcal{N}(f(\mathbf{x}_n, \nu))$
    - $\nu$ is the variance of noise introduced into the function observations.

<br>

- The prior and these data induce a poterior over functions: **acquisition function**

- We denote by $$a: \mathcal{X} \rightarrow \mathbb{R}^+$$ 
    - determines what point in $\mathcal{X}$ should be evaluated next via a proxy optimization: $$\mathbf{x}_{next} = \arg \max_{\mathbf{x}} a(\mathbf{x})$$ 
    
    - several  different functions have been proposed.

- In general, these acquisition functions depend on the previous observations, as well as the <u>GP hyperparameters</u>; 


    - We denote this dependence as $$a(\mathbf{x} ; \{ \mathbf{x}_n , y_n \}, \theta)$$




<br>

- There are several popular choices of acquisition function. 
- Under the Gaussian process prior, these functions depend on the model solely through its 
    - predictive mean function, $\mu(\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)$,
    - predictive variance function, $\sigma^2 (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)$




<br>

- In the proceeding, we will denote the best current value as $$\mathbf{x}_{best} = \arg\min_{\mathbf{x}_n} f(\mathbf{x}_n),$$
- $\Phi(\cdot)$ will denote the cumulative distribution function of the standard normal,
- $\phi(\cdot)$ will denote the standard normal density function.


<br>

---

### 2.2.1. Probability of Improvement

- **One intuitive strategy**: to maximize the probability of improving over the best current value, $\mathbf{x}_{best}$. 

> - 현재까지 관찰된 최소값(minimum value)보다 더 나은 값이 나올 확률을 계산
> - 이전 최소값보다 작을 확률을 계산하기 위해서는 손실 함수 값이 평균과 표준 편차로 정의된 가우시안 분포에서 얼마나 작은 값인지 계산
> -개선 확률은 현재의 최적 해보다 더 좋은 해가 존재할 확률을 나타내며, 이 확률을 최대화하는 방향으로 다음 검색 지점을 선택

- Probability of improvement (PI)는, 특정 지점의 함수 값이 지금 best 함수 값인 $\mathbf{x}_{best}$ 보다 작을 확률을 사용

    - Estimated function $f(\mathbf{x})$의 값은 정해진 값이 아니라 확률 값이기 때문에, PI는 $\mathbf{x}$에서의 $u(\mathbf{x})$의 expectation으로 표현된다.
    - 이때 $\mathcal{N}(f;\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}))$는 mean function $\mu(\mathbf{x})$와 kernel function $k(\mathbf{x}, \mathbf{x})$로 표현되는 normal distribution이고, $\Phi(\cdot)$은 cdf를 의미

    - 아래 그림에서 이미 explore가 많이 된 지점이 PI가 높음

- Under the GP, this can be computed analytically as 

$$a_{PI} (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta) = \Phi(\gamma(\mathbf{x}))$$

$$\gamma(\mathbf{x}) = \dfrac{f(\mathbf{x}_{best}) - \mu(\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)}{\sigma(\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)}$$

- 이 수식들은 PI를 계산하는 방법을 보여줍니다.

    - 첫 번째 수식은 PI를 계산하는 식입니다: 여기서, $a_{PI}$는 Probability of Improvement를 나타내며, $\mathbf{x}$는 문제 공간의 한 지점을 나타냅니다. $\Phi$는 표준 정규 분포의 누적 분포 함수(cumulative distribution function, CDF)를 나타냅니다. $\gamma(\mathbf{x})$는 함수 $f$의 값이 현재 최적의 값 $f(\mathbf{x}_{best})$보다 높을 확률을 계산하는 데 사용되는 표준화된 함수입니다.

    - 두 번째 수식은 $\gamma(\mathbf{x})$를 계산하는 방법을 보여줍니다: 여기서, $f(\mathbf{x}_{best})$는 현재 최적의 값을 나타내며, $\mu(\mathbf{x}; { \mathbf{x}_n , y_n }, \theta)$는 가우시안 프로세스 모델에 의해 예측된 평균 함수 값을 나타냅니다. $\sigma(\mathbf{x}; { \mathbf{x}_n , y_n }, \theta)$는 가우시안 프로세스 모델에 의해 예측된 표준 편차 값을 나타냅니다.

    - 따라서, 첫 번째 수식은 $\gamma(\mathbf{x})$를 사용하여 PI를 계산하는 방법을 보여주고, 두 번째 수식은 $\gamma(\mathbf{x})$를 계산하는 방법을 보여줍니다. 이 두 수식을 함께 사용하여 문제 공간의 각 지점에서 개선될 확률을 계산하고, 그 중 최대 개선 확률을 가진 지점을 다음 검색 지점으로 선택할 수 있습니다.






<div style="text-align: center;">
  <img src="./img/1-1.png"  alt="이미지 설명" style="width: 50%; height: auto;">
</div>





<br>

### 2.2.2. Expected Improvement

- PI의 가장 큰 문제점 중 하나는, ‘improvement’ 될 수 있는 확률만 보기 때문에, 확률이 조금 더 낫을지라도, 궁극적으로는 더 큰 improvement가 가능한 point를 고를 수 없다는 점

- 다시 말하면 exploit에 집중하느라 explore에 취약하다는 단점이 있다.

- Expected improvement (EI)는 utility function을 0, 1이 아니라, linear 꼴로 정의하기 때문에 그 차이를 반영할 수 있다.

- 주의할 점은, EI가 PI의 expectation이 아니라는 점이다. 그냥 이름만 비슷한거고 완전히 다른 function이라고 생각하면 된다. PI와 마찬가지로 EI역시 u(x)의 expectation을 계산해야 한다.

- EI를 그림으로 나타내면 다음과 같다. PI처럼 이미 explore가 많이 된 곳을 또 찾는 실수는 덜 저지른다는 것을 볼 수 있다.

<div style="text-align: center;">
  <img src="./img/1-2.png"  alt="이미지 설명" style="width: 50%; height: auto;">
</div>

- Alternatively, one could choose to maximize the expected improvement (EI) over the current best.

- This also has closed form under the Gaussian process: 
$$a_{EI} (\mathbf{x} ; \{ \mathbf{x}_n  , y_n \}, \theta) = \sigma (\mathbf{x} ; \{ \mathbf{x}_n  , y_n \}, \theta ) (\gamma(\mathbf{x}) \Phi(\gamma(\mathbf{x})) + \mathcal{N}(\gamma(\mathbf{x}); 0, 1))$$

- EI는 주어진 하이퍼파라미터 설정에서 목표 함수의 값이 현재까지 발견된 최적 값보다 얼마나 개선될 것으로 기대되는 정도를 측정

- 탐색-활용(Exploration-Exploitation) 트레이드오프를 관리하는 데 도움이 되며, 현재까지 알려진 최적의 지점 근처에서 더 나은 값을 찾거나, 아직 불확실한 영역을 탐색하는 데 사용



- 이 수식은 주어진 하이퍼파라미터 설정 $\mathbf{x}$에 대한 Expected Improvement 값을 계산하는 방법을 보여줍니다. 여기서 $\theta$는 서로게이트 모델(가우시안 프로세스)의 파라미터, ${ \mathbf{x}_n, y_n }$은 현재까지 관측된 하이퍼파라미터 설정과 목표 함수 값들입니다.

- 수식의 각 요소는 다음과 같은 의미를 가집니다:

    - $\sigma (\mathbf{x} ; { \mathbf{x}_n , y_n }, \theta )$: 예측된 불확실성을 나타냅니다. 불확실성이 큰 지점에서는 탐색이 이루어질 가능성이 높습니다.
    - $\gamma(\mathbf{x}) \Phi(\gamma(\mathbf{x}))$: 목표 함수 값이 현재 최적의 값보다 클 확률을 나타냅니다.
    - $\mathcal{N}(\gamma(\mathbf{x}); 0, 1)$: 확률밀도함수를 통해 현재 지점의 확률적 특성을 나타냅니다.

- 이 수식을 통해 각 하이퍼파라미터 설정에 대한 Expected Improvement 값을 계산할 수 있으며, 가장 큰 Expected Improvement 값을 가진 하이퍼파라미터 설정을 선택합니다. 이렇게 선택된 하이퍼파라미터 설정은 다음 단계에서 머신러닝 모델의 성능 평가에 사용됩니다.

<br>

### 2.2.3. GP Upper Confidence Bound

- 이 방법은 주어진 하이퍼파라미터 설정에 대한 목표 함수의 예측 평균과 불확실성을 동시에 고려하여 탐색-활용 트레이드오프를 관리합니다. UCB는 예측 평균과 불확실성에 대한 가중치를 조절하는 하이퍼파라미터인 $\kappa$를 사용합니다.

- A more recent development is the idea of exploiting lower
confidence bounds (upper, when considering maximization) to construct acquisition functions that minimize regret over the course of their optimization

- 

$$a_{LCB} (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta) = \mu (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta) - \kappa \sigma (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)$$


- 이 수식의 각 요소는 다음과 같은 의미를 가집니다:

    - $\mu (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)$: 가우시안 프로세스에서 예측된 평균입니다. 이 값은 현재 하이퍼파라미터 설정에 대한 목표 함수 값의 추정치를 나타냅니다.
    - $\sigma (\mathbf{x}; \{ \mathbf{x}_n , y_n \}, \theta)$: 예측된 표준편차로, 현재 하이퍼파라미터 설정에 대한 불확실성을 나타냅니다.
    - $\kappa$: 탐색-활용 트레이드오프를 조절하는 하이퍼파라미터입니다. 값이 클수록 불확실성이 높은 영역에 더 많은 가중치를 부여하고, 값이 작을수록 예측 평균에 더 많은 가중치를 부여합니다.

- Form도 간단하고, 조절하기 쉽기도 하지만, hyperparameter를 또 조정해야한다는 문제 때문에 이 논문에서는 다루지 않는다.

<div style="text-align: center;">
  <img src="./img/1-3.png"  alt="이미지 설명" style="width: 50%; height: auto;">
</div>

---









- 저자는 예상 개선(Expected Improvement, EI) 기준에 중점을 둘 것이라고 언급하고 있습니다. EI는 개선 확률(Probability of Improvement)보다 더 낫게 작동하는 것으로 나타났습니다. 또한 GP-UCB와는 달리, 별도의 튜닝 매개변수가 필요하지 않다는 장점이 있습니다.


<br>
<br>

# 3.  Practical Considerations for Bayesian Optimization of Hyperparameters.

- 😑 Bayesian optimization은 굉장히 impractical하다. 여러가지 이유가 있는데, 크게는 다음과 같은 이유들이 있다. 

    - Hyperparameter search를 하기 위해 BO를 사용하는데, BO를 사용하기 위해서는 GP의 hyperparameter들을 튜닝해야한다 (kernel function의 parameter 등)
    - 어떤 stochastic assumption을 하느냐에 따라 (어떤 kernel function을 사용해야할지 등) 결과가 천차만별로 바뀌는데, (model selection에 민감한데) 어떤 선택이 가장 좋은지에 대한 가이드가 전혀 없다.
    - Acquisition function을 사용해 다음 지점을 찾는 과정 자체가 sequential하기 때문에 grid search나 random search와는 다르게 parallelization이 불가능하다.
    - 위에 대한 문제점들이 전부 해결된다고 하더라도 software implementation이 쉽지 않다.


<br>



- 😀 이런 문제점들을 해결하기 위해 이 논문은 먼저 kernel function을 여러 실험적 결과 등을 통해 Matern 5/2 kernel이 가장 실험적으로 좋은 결과를 낸다는 결론을 내린다 (즉, kernel function은 언제나 Matern 5/2를 쓰면 된다). 
- 또한 acquisition function도 EI로 고정한다. 
- 다음으로 GP의 hyperparameter들을 Bayesian approach를 통해 acquisition function을 hyperparameter에 대해 marginalize한다. 이 marginalized acquisition function은 (integrated acquisition function이라고 한다) MCMC로 풀 수 있다. 

- 마지막으로 이 논문은 이론적으로 tractable한 Bayesian optimization의 parallelized version을 (MCMC estimation이다) 제안한다.




---
 
<br>

- 하이퍼파라미터의 베이지안 최적화에 대한 실용적인 고려 사항에 대해 설명하고 있습니다. 비록 비용이 많이 드는 함수를 최적화하는 우아한 프레임워크지만, 몇 가지 한계로 인해 머신러닝 문제에서 하이퍼파라미터 최적화를 위한 널리 사용되는 기법이 되지 못했습니다. 
- 이러한 한계는 공분산 함수와 관련된 하이퍼파라미터의 선택, 함수 평가 시간, 다중 코어 병렬성 활용 등에 관련되어 있습니다. 이 문단에서는 이러한 문제에 대한 해결책을 제안하고 있습니다.

    - First, it is unclear for practical problems what an appropriate choice is for the covariance function and its associated hyperparameters.

    - Second, as the function evaluation itself may involve a time-consuming optimization procedure, problems may vary significantly in duration and this should be taken into account.

    - Third, optimization algorithms should take advantage of multi-core parallelism in order to map well onto modern computational environments.






## 3.1. Covariance Functions and Treatment of Covariance Hyperparameters.

- **Automatic relevance determination (ARD) squared exponential kernel** is often a default choice for Gaussian process regression:


$$K_{SE}(\mathbf{x}, \mathbf{x'}) = \theta_0 \exp\{ -\dfrac{1}{2} r^2 (\mathbf{x}, \mathbf{x'}) \}$$

$$r^2(\mathbf{x}, \mathbf{x'}) = \sum^D_{d = 1} (x_d - x'_d)^2 / \theta^2_d$$




- However, sample functions with this covariance function are unrealistically smooth for practical optimization problems.

- 가장 많이 쓰이는 Squared-exponential function의 가장 큰 문제는 ‘smoothness’로, 복잡한 모델을 표현하기에는 너무 ‘smooth’한 function만 estimate할 수 있다는 단점이 있다.


<br>

- 이를 해결하기 위해 이 논문에서는 Matern kernel function을 사용하며, 특히 그 hyperparameter로 5와 2를 사용하는 Matern 5/2를 사용하고 있다. 
- 실제로 structured SVM의 hyperparameter를 찾을 때 여러 kernel function 중에서 가장 좋은 kernel이 무엇인지 아래와 같은 실험들 끝에 얻은 결과이다.

- We instead
propose the use of the **ARD Matern 5/2 kernel**:
$$K_{M52} (\mathbf{x}, \mathbf{x'}) = \theta_0(1 + \sqrt{5r^2(\mathbf{x}, \mathbf{x'})} + \dfrac{5}{3}r^2 (\mathbf{x}, \mathbf{x'}) )\exp\{ -\sqrt{5r^2 (\mathbf{x}, \mathbf{x'}) } \}$$

- This covariance function results in sample functions which are twice differentiable, an assumption that corresponds to those made by, e.g., quasi-Newton methods, but without requiring the smoothness of the squared exponential.

- 이 GP의 hyperparameter는 $\theta_0, \theta_d$로, d가 1부터 D까지 있으니 총 D+1 개의 hyperparameter를 필요로 한다.


<br>

---

- After choosing the form of the covariance, we must also manage the hyperparameters that govern its behavior, as well as that of the mean function.

- 이제 covariance의 형태를 결정했으니, GP의 hyperparameter를 없애는 일이 남았다.

- 우리가 optimize하고 싶은 hyperparameter의 dimension이 D라고 해보자

- 이때 GP의 hyperparameter의 개수는 D+3개가 된다:
    - 바로 앞에서 언급한 D+1개와, constant mean function의 값 $m$, 그리고 noise $\nu$

<br>


- (이 논문에서는 <u>hyperparameter를 완전하게 Bayesian으로 처리하기 위하여</u> 모든 hyperparameter $\theta$ (D+3 dimensional vector)에 대해 acquisition function을 marginalize한 다음에, 다음과 같은 integrated acquisition function을 계산하는 방법을 제안)

- The most commonly advocated approach is to use a point estimate of these parameters by optimizing the marginal likelihood under the Gaussian process:

$$p(\mathbf{y} | \{ \mathbf{x} \}^N_{n=1}, \theta, \nu, m ) = \mathcal{N} (\mathbf{y} | m\mathbf{1}, \boldsymbol{\Sigma}_{\theta} + \nu \mathbf{I}),$$ 

- where  $\mathbf{y} = [y_1, y_2, \cdots, y_n]^\top$, and $\boldsymbol{\Sigma}_{\theta}$ is the covariance matrix resulting from the $N$ input points under the hyperparameters $\theta$.


- However, for a fully-Bayesian treatment of hyperparameters (summarized here by $\theta$ alone), **it is desirable to marginalize over hyperparameters and compute the <u>integrated acquisition function</u>**:

$$\hat{a}(\mathbf{x}; \{\mathbf{x}_n , y_n \}) = \int a(\mathbf{x} ; \{ \mathbf{x}, y_n \}, \theta) p (\theta | \{ \mathbf{x}_n, y_n \}^N_{n=1}) d\theta$$

- where $a(\mathbf{x})$ depends on $\theta$ and all of the observations. 

<br>

- PI와 EI에 대해서는 이 integrated acquisition function을 계산하기 위해 다양한 GP hyperparameter에 대한 GP posterior를 계산한 다음, integrated acquisition function의 Monte Carlo estimatation을 구하는 것이 가능.


- We can therefore blend acquisition functions arising from samples from
the posterior over GP hyperparameters and have a Monte Carlo estimate of the integrated expected improvement.

    - These samples can be acquired efficiently using slice sampling, as
described in Murray and Adams (2010)


- 베이지안 최적화에서 하이퍼파라미터 불확실성을 고려하기 위해, 가우시안 프로세스의 하이퍼파라미터에 대한 사후 분포에서 나온 샘플을 이용하여 취득 함수를 결합하고, 몬테카를로 방법을 사용하여 통합된 예상 개선치를 추정할 수 있습니다. 이를 위해 slice sampling 방법을 사용할 수 있습니다. 최적화와 마르코프 체인 몬테카를로 방법은 계산 비용이 크기 때문에 완전 베이지안 처리가 현실적이며, Figure 1은 통합된 예상 개선치가 취득 함수를 어떻게 변화시키는지 보여줍니다.


<div style="text-align: center;">
  <img src="./img/1-5.png"  alt="이미지 설명" style="width: 50%; height: auto;">
</div>





