# 배운 것 정리!

## 1.1. Tensor Decomposition?

간단한 Tensor들에 대해서 연산들의 조합으로 하나의 Tensor 표현 

### Singular Value Decomposition(SVD)

SVD는 low-rank approximations를 계산하는데 사용된다!

rank k는 k개만큼 잘라서 본다!!

## 2.1. Basic notation

Matrix version(2-way)

**Tensor Version(3-way)**

Tensor Version(d-way)

## 2.2. CP decomposition

* Detecting low-rank 3-way structure

weight matrix를 decomposition한다!

-> conv weight x -> low rank M으로 approximation한다!

-> M은 rank 1 vector들의 합으로 표현된다!

![쇼쇼숏](https://user-images.githubusercontent.com/59636424/144700878-83baf0ac-bfe5-43ab-8090-76257309455b.PNG)

### Fitting CPLhardships

Rank 계산은 NP-Hard 문제이다!!

column들이 linearly dependent 할 수 있다.

CP는 permutation과 scaling에 대해서 unique한 성질을 가지고 있다.

### Fitting CP: ALS(Alternating Least Square)

3차원 텐서의 경우, Component가 3개이므로 하나를 계산할 때, 나머지 2개를 고정시킨다. -> 하나에 대해서 list scale문제를 푼다!!

-> 그렇게 각각 3단계를 거친다!!!!

아래와 같이, 계산 시에는 matrix로 표현한다!!!

![ㅕㅕㅕㅕ](https://user-images.githubusercontent.com/59636424/144701067-92fc05b6-e783-44b5-b9e6-52f0d0b75551.PNG)

* CP & Tucker decomposition

Tucker는 core tensor로 이뤄짐

![ㅛㅛㅕㅛㅕㅛㅕㅛ](https://user-images.githubusercontent.com/59636424/144701309-411f0dd5-a595-4178-92a8-4fb8e2bc3a30.PNG)

CP는 core tensor가 각 가중치들의 diagonal하게 나열된 tensor가 된다.

![ㅛㅕㅛㅕㅛㅕㅛㄹㅇㄹㄿ](https://user-images.githubusercontent.com/59636424/144701324-a6ef99cc-c4c3-4db0-9f3f-79c93258cd16.PNG)


