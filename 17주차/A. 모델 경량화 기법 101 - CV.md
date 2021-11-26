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

-> LTH가 잘 됨!!

### Linear Mode Connectivity and the Lottery Ticket Hypothesis(ICML 2020)

네트워크 학습과 수렴 관련된 실험!!


