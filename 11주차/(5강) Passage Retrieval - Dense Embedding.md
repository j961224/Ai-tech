# 1. Passage Retrieval - Dense Embedding 정리!

## 1. Introduction to Dense Embedding

### 1-1. Passage Embedding

구절을 벡터로 변환하는 것!

### 1-2. Sparase Embedding

벡터 크기는 크지만 0이 아닌 숫자는 적다!

### 1-3. Sparase Embedding의 제한

* **차원의 수가 매우 크므로 compressed format으로 극복!**

=> non-zero 값만 저장하는 방법

**sparse embedding은 유사성을 고려하지 못 하는게 큰 단점!!!**

### 1-4. Dense Embedding이란?

고밀도 벡터로 mapping이 되므로 Sparse Embedding과 다르게 길이가 크지 않다! (길이: 50~1000) => non-zero값이 대부분!

**각 차원이 모두 합쳐져서 벡터 공간 상에서 위치를 나타낸다!!**

![rr](https://user-images.githubusercontent.com/59636424/137051138-0dfb05c2-1f9a-4e52-b8a3-a399b0c49706.PNG)

위 사진은 sparse embedding -> dense embedding!

### 1-5. Retrieval: Sparse vs Dense

sparse는 단어 존재 유무는 맞추기 쉽지만 의미를 맞추기 쉽지 않다.

=> dense는 반면에 의미를 맞춘다!!

|Sparse Embedding|Dense Embedding|
|----|----|
|중요한 term들이 정확히 일치하는데 성능이 좋음|단어 유사성을 파악하는데 성능이 좋음|
|임베딩 구축 시, 추가적 학습 불가능!|추가적인 학습 가능!|

=> Sparse와 Dense 둘 다 사용되는 경우가 점점 생기는 중!

### 1-6. Dense Embedding의 Passage Retrieval의 Overview

![eeeeeeee](https://user-images.githubusercontent.com/59636424/137051866-602fcb41-5e5b-47ef-a618-5223e0800023.PNG)

BERT_Q -> Question에 해당하는 encoder -> Question에 해당하는 encoder가 sentence encoding하여 CLS vector로 h_q를 내보냄!

BERT_b -> Passage는 다른 형태의 파라미터를 이용해서 CLS를 이용해 h_b를 내보냄!

**h_q와 h_b 벡터는 같은 사이즈여서 유사도를 측정 가능!**

==> **BERT_Q와 BERT_B는 Dense embedding을 생성한 encoder 훈련!!**

## 2. Dense Encoder 훈련

### 2-1. Dense Encoder란?

MRC는 passage와 question을 input으로 준 반면, **question과 passage를 각각 embedding을 구하기 위해 독립적으로 넣음!**

**Embedding을 output으로 내는 것이 목적이기 때문에 CLS token의 최종 embedding을 output!**

### 2-2. Dense Encoder 학습 목표와 학습 데이터

* training 목표: **question과 passage dense embedding 거리 줄이기!** -> inner product 높이기!
* 어려운 점: Question & Passage pair 찾기! => 그래서 기존 MRC 데이터셋 이용!

### 2-3. Negative Sampling

관련 없는 question과 passage간의 embedding 거리는 멀게 함 -> negative

반대로, 연관된 question과 passage간의 embedding 거리는 좁힘! -> positive

* negative examples 뽑는 법!

: Corpus 내에서 랜덤하게 뽑기!, **좀 더 헷갈리는 negative sample 뽑기(TF-IDF 스코어는 높지만 답을 포함하지 않는 샘플)**

### 2-4. Objective function

NLL loss 사용! => **positive passage와 question 간에 유사도 score(높을수록 유사도 좋음)와 negative sample의 score로 softmax한 값의 확률 값을 negative log likelihood에 적용!**

![zzzzz](https://user-images.githubusercontent.com/59636424/137052971-840eaa83-6945-4869-a45d-27517dff5f54.PNG)

Positive Passage와 Negative Passage로 나누면, poisitve passage score 분자로 전체 passage 분모로 두게하여 계산!

### 2-5. Dense Encoder 측정 방법

**Top-k retrieval accuracy**: retrieve된 passage 중에, 답을 포함한 passage 비율!

## 3. Dense Encoder의 Passage Retrieval

dense encoding -> retrieval하는 방법

Query와 Passage embedding한 것에, Query로부터 가까운 Passage의 순위를 측정!!

### 3-1. retrieval -> open-domain question answering

![rrr](https://user-images.githubusercontent.com/59636424/137053584-f156c12b-eca6-46a9-9b9a-6eab2b896924.PNG)

이렇게 가까운 Passage로 MRC 모델에 넣어 사용 가능하다.

### 3-2. Dense encoding 개선 방법

학습 개선 방법

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) 참조


# 2. 실습 정리!

## 1. BERT를 활용한 Dense Passage Retrieval (In-batch) 실습!

* **in-batch negative 사용** 

현재 question과 example passage의 유사도 score를 최대화시키면서, 현재 question과 다른 example passage 유사도 score 최소화

* question과 query 분리

```python
q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
```


* Bert encoder를 통한 query와 question의 pooler output을 뽑기 위한 class 정의

```python
class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, attention_mask=None, token_type_ids=None): 
  
      outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
      
      pooled_output = outputs[1]

      return pooled_output
```

* train 함수 부분의 similiarity score 계산 및 target 설정

```python
# Calculate similarity score & loss
# (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))

# target: position of positive samples = diagonal element -> 대각선으로 각각 진짜 question과 context 쌍의 유사도 score가 위치해있다.
targets = torch.arange(0, args.per_device_train_batch_size).long()

sim_scores = F.log_softmax(sim_scores, dim=1)

loss = F.nll_loss(sim_scores, targets)
```


