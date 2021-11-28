# 1. 배운 것 정리!

## 1. 주요 경량화 기법

> * 효율적인 architecture(MobileNet 등 + NAS)
> * Pruning(Structued/Unstructured)
> * Knowledge Distillation(Response based 등)
> * Weight Factorization(Tucker decomposition 등)
> * Quantization



## 2. Pruning

중요도가 낮은 파라미터 제거!!

* 어떤 단위로 Pruning?!

Structured(group)/Unstructured(fine grained)

* 어떤 기준으로 Pruning?!

중요도 정하기(Magnitude(L2, L1), BN 등!)

* 어떻게 기준 적용?!

Network 전체로 줄 세워서(global), Layer 마다 동일 비율로 기준!(Local)

* 어떤 phase에?!

학습된 모델에 / Initialize 시점에(학습 전에 pruning하고 시작)

## 2.1. Structured Pruning

그룹단위로 pruning!(channel/filter/layer level 단위로!)

### Learning Efficient Convolutional Networks through Network Slimming 논문!

Structured(group) Pruning

BN scaling factor 기준으로 Pruning

Global 기준으로 적용

학습된 모델에 적용

* Overview

BatchNorm에 scaling parameter

ImageNet 기준, VGG11, 50% pruning, FLOPs 30.4%, 파라미터 수 82.5% 감소!

* Scaling factor 감마

**감마가 크다면, 해당 filter weight 크기는 일반적으로 클 것!**

**감마 기준으로, p% channel을 pruning!**

-> 값이 큰 애들은 살리고 작은 애들은 죽이고 Regularization 적용!

### HRank: Filter Pruning sing High-Rank Feature Map (CVPR 2020) 논문!

Structured(group) Pruning

Feature map Rank(SVD) 기준으로 Pruning

Local 기준으로 적용

학습된 모델에 적용

* Overview

Feature map output에 적용!

ResNet50, ImageNet 기준, 약 44%의 FLOPs 감소와 1.17% 성능 하락!

-> 이미지를 넣으면 각 Filter마다 전부 SVD 적용!! -> Rank 계산

* Feature map output

이미지에 따라, Feature map output은 당연히 달라짐! -> 그때마다, SVD Rank 개수가 달라지는 것 아닌가?!

-> 각 Batch로 Feature map output을 계산 -> Rank를 구했을 때, 차이가 없음을 실험적으로 보임!

* Rank 계산 과정

![ㄷㄱㄷㄱㄷㄱ](https://user-images.githubusercontent.com/59636424/143514244-82158268-3aa8-492d-a2b5-1c1c3eb12eb1.PNG)

-> TORCH.MATRIX_RANK 사용!

### 2.2. Unstructed Pruning

파라미터 각각 독립적으로 Pruning -> 네트워크 내부의행렬이 점점 희소!

### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks(ICLR 2019) 논문!

Unstructured Pruning(fine grained)

L1 norm 기준으로 Pruning

Global기준으로 적용

학습된 모델에 적용 -> 아주 약간 pruning하고, 재학습 반복

* Overview

LotteryTicket Hypothesis -> Dense, randomly-initalized, feed-forward net은 기존의 network와 필적하는 성능을 갖는 sub networks를 갖는다.

* Identifying winning tickets

> * 1. 네트워크를 임의로 초기화!
> * 2. 네트워크를 j번 학습하여 파라미터 세타_j를 도출
> * 3. 파라미터 세타_j를 p%만큼 pruning하여 mask m을 생성!(p는 보통 20%)
> * 4. Mask되지 않은 파라미터를 세타_0으로 되돌리고, 이를 winning ticket이라 지칭!
> * 5. Target sparsity(e.g. 90%)에 도달할때 까지 2~4반복

* Contribution

Pruning -> Fine-Tuning 방식이 아닌, 완전 초기치로 되돌려서 처음부터 training함!!

### The Lottery Ticket Hypothesis: Weight Rewinding(arXiv 2019) 논문!

초기치로 되돌리지 않고, k번째 epoch에서 학습한 파라미터로 네트워크를 초기화하는 것으로 학습 안정화를 보임!

* Weight Rewinding

![ㅈㄷㅈㄷㅈㄷㅈ](https://user-images.githubusercontent.com/59636424/143515067-c127a737-f3e8-401a-bb8b-d8f937f844b1.PNG)

### Comparing Rewinding And Fine-tuning in Neural Network Pruning(ICLR 2020): Learning Rate Rewinding

앞에서 k를 뭐를 잡을지 결정! -> Weight rewinding이 아닌, Learning rate scheduling을 k 시점으로 rewinding 하자!

* LR Rewinding

![ㄷㄱㄷㄱㄷㄱㄷㄱㄷㄱ](https://user-images.githubusercontent.com/59636424/143515278-a83b6f5c-7620-4c28-96d5-9f6bc5bd62cf.PNG)

-> Weight와 lr을 Rewind하는 Weight Rewinding이 아닌, lr만 rewind하는 방법이다!

-> LTH가 잘 됨!!

### Linear Mode Connectivity and the Lottery Ticket Hypothesis(ICML 2020)

네트워크 학습과 수렴 관련된 실험!!

-> 초기치 weight에서 다른 seed를 가지고 학습시키기! -> 수렴되는 위치가 다를 것이다.

이 두 weight 공간 사이의 interploated net의 성능을 확인!!

### Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask (neurlPS 2019)

LTH(Lottery Ticket)은 final weight에 L1 norm을 줄 세워서 masking하는데 inital weight를 고려하면?!

![tttttt](https://user-images.githubusercontent.com/59636424/143763743-5882845c-20e5-4acf-9f2a-0b6400a3070d.PNG)

-> final weight을 기준으로, 자르는데 그것보다 큰 애들은 살리고 나머지 죽이기!

![ttty](https://user-images.githubusercontent.com/59636424/143763803-a9ab6803-f033-4909-be62-ec1023544a9d.PNG)

### 2.3. Pruning: overall

> * Pruning?
>> * 중요도가 낮은 파라미터를 제거하는 것
> * 어떤 단위로 Pruning?
>> * Structured(group)/Unstructured(fine grained)
> * 어떤 기준으로 Pruning?
>> * 중요도 정하기(Magnitude(L2, L1), BN scaling factor, Energy-based, Feature map..)
> * 기준은 어떻게 적용?
>> * Network 전체로 줄 세워서(global), Layer마다 내부로 기준(local)
> * 어떤 phase에?
>> * 학습된 모델에(trained model) / initalize 시점에(pruning at initalization)

### 2.4. Pruning: pruning at initalization

* Pruning at initialization(unstructured)

SNIP: Training 데이터 샘플, Forward해서 Gradient와 Weight의 곱의 절대값으로!

GraSP: Training 데이터 샘플, Forward해서 Hessian-gradient product와 Weight의 곱으로!

SynFlow: 전부 1로 된 가상 데이터를 Forward해서 Gradient와 Weight을 곱으로!

* Intuition behind(끼워 맞추기)

k번째 prune 되었을 때와 안 했을 때 Loss 차이가 아래 수식과 비례! -> 크면, 해당 파라미터의 기여가 크다 / 작으면, 해당 파라미터의 기여가 작다

![re](https://user-images.githubusercontent.com/59636424/143764340-12bc2c50-6c90-4557-89af-126350979541.PNG)

* Experiment

![ttttt](https://user-images.githubusercontent.com/59636424/143764384-370da5b9-7d4a-4ced-9395-fa1ad6db0e98.PNG)

-> SynFlow(Synatic Flow)라는 것을 유지하는데, compression이 smooth하게 떨어진다!

### 2.5. Pruning: pruning at initialization on NAS

떡잎을 본다! -> 네트워크의 가능성을 본다! -> Training Free NAS/AutoML

* Short Summary

Pruning at initialization의 기법들을 일종의 score로 network를 정렬

-> 각 기법의 score와 실제 학습 결과의 상관 관계를 확인 -> 생각보다 상관 계수가 높고 voting하면 더욱 높다!

**학습이 필요없는, 간접적으로 모델을 평가하는 기법으로써 활용이 될 가능성이 높다!**

* pruning at initalization이 NAS랑 연결된다.

### 3.0. Knowledge Distillation

**teacher 정보를 어떻게 빼내는가?**

> * Feature-based knowledge
> * Relation-Based knowledge
> * Response-Based Knowledge

### 3.1. Response-Based Knowledge Distillation

last output layer를 쓰면서, 직접적인 final prediction을 활용!

### 3.2. Feature-Based Knowledge Distillation

layer 중간 중간의 feature를 student가 학습하도록 한다!

![rtrtrtr](https://user-images.githubusercontent.com/59636424/143765149-dbacb69f-82f3-4295-9405-6123acfbde49.PNG)

-> 중간중간, output결과가 비슷하도록 학습!!!

-> **목적은 student feature가 teacher feature와 유사해지도록 한다!**

논문의 주된 방향은, **유용한 정보는 최대로 가져오고, 중요하지 않은 정보는 가져오지 않도록 한다!**

-> 중간 결과를 가져오니, network 구조에 매우 dependent한다!

### 3-3. Relation-Based Knowledge Distillation

다른 layer나 Sample간의 관계를 정의하여 knowledge distillation 수행!

* Response-based: Final prediction을 활용
* Feature-based: Intermediate representation을 활용
* Relation-based: Data sample들 또는 Feature map output들 간의 관계를 학습

-> 전반적으로 Architectural Dependency가 높음!

* Noisy Student Training

Teacher model을 labeled된 데이터로 학습 ->  sudo label로 unlabel였던 데이터를 학습 -> Data 및 model noise를 가지고 학습!

