# 1. Reducing Training Bias 정리!

## 1. Bias 정의

학습할 때 과적합을 막거나 사전 지식을 주입!!

### 1-1. Gender Bias

특성 성별과 행동을 연관하여 예측하는 오류!!

ex) cooking 시, 여자라고 예측하는 경우

### 1-2. Sampling Bias

random하지 않고, 편향되게 샘플링하여 왜곡될 수 있다.

ex) 중산층 이상으로만 표본

## 2. ODQA에 Bias

### 2-1. Reader model의 훈련 bias

한정된 데이터셋만 학습이 되면, 정답을 제대로 못 낼 것이다!(문서에 대한 독해 능력 하락)

### 2-2. Training Bias 해결 방법!

* negative example 보여주고 Train하기

: negative를 잘 구별할 수 있도록 잘못된 예시를 보여주기

* no answer bias 추가

: 답이 문서에 없다면, no answer 줄 수 있도록 학습하기!

=> no answer bias 추가하기!!!! => **시퀀스 길이 외에 1개의 token이 더 있다 생각하기!**

=> 훈련 모델 마지막 레이어에 weight에 훈련 가능한 bias를 하나 더 추가! 

=> Softmax 시, 최종적으로 no answer bias 위치의 확률이 높다면(start, end 확률) no answer 대답!

### 2-3. Train negative examples

어떻게 좋은 Negative sample을 만들 수 있나???

* Corpus 내에 랜덤해서 뽑지만 좀 더 헷갈리는 negative 샘플을 뽑기

: 답변같지만, 답변이 아닌 샘플 가져오기!

## 3. Dataset에 Annotation Bias

데이터 제작 시, 생기는 issue!!

### 3-1. Annotation bias란?

질문하는 사람이 답을 알고 있는 상태로 질문하는 편향!!

-> 그래서, 질문이 쉬워지는 경우가 생긴다.

TrivaQA, SQuAD 데이터는 좀 심하다.

### 3-2. Annotation bias 영향

데이터 셋 별로, bias가 생길 수 있다.

SQuAD에서는 BM25가 DPR보다 성능이 잘 나오는 경우가 있다. (보통, DPR이 성적이 더 높음)

=> SQuAD의 annotation bias가 단어의 overlap을 유도하므로, BM25에 유리한 setup이다!

(BM25 + DPR이 꽤 괜찮은 성적이 나왔음)

### 3-3. Annotation bias 다루기

bias를 고려해서 데이터를 모아야함!

### 3-4. Another bias

SQuAD는 Passage가 주어지고, 그 내에서 질문과 답을 생성!

=> ODQA에 applicable하지 않는 질문들이 존재한다.


