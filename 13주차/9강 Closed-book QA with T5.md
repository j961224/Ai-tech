# 1. 배운 것 정리!

## 1. Closed-book Question Answering

![ㄱㄱㄱㄱㄱ](https://user-images.githubusercontent.com/59636424/140919198-95884bee-94c7-4e1f-8ea8-19c84c136375.PNG)

현재까지 해당 문서를 찾아서 모델이 답을 찾는 방식이었다!

* Idea of Closed-book Question Answering

: 사전학습 언어모델 자체가 이미 하나의 knowledge storage가 있어서 그 안에서 답을 가져온다!

![ㄳㄳㄳㄳ](https://user-images.githubusercontent.com/59636424/140919444-3dc87132-b77e-45a7-8fec-0dfa6f83e34f.PNG)

=> 대량의 지식을 학습한 사전학습된 언어 모델이니 이 모델을 knowledge storage라고 볼 수 있다!!

* Zero-shot QA performance of GPT-2

: 사전학습 시에 전혀 본 적 없는 것도 대답이 가능!!

* Open-book QA vs Closed-book QA

![ㅈㅈㅈㅈ](https://user-images.githubusercontent.com/59636424/140919825-c8dfee84-a755-4eb3-8490-76526e29e43d.PNG)

## 2. Text-to-Text Format

Closed-book QA 방법은 Generation-based MRC 방법과 유사

-> 단, **질문만 들어간다!**

-> BART와 같은 seq-to-seq 형태의 Transformer 모델을 사용!

* Text-toText Format

: input을 text로 받아서 output을 text로 생성한다!

* Model Overview

![ㄱㄷ](https://user-images.githubusercontent.com/59636424/140922132-1163c6e8-341b-4264-9465-89f04f66d912.PNG)

### 2-1. T5

![ㄱㄱㄱㄱ'](https://user-images.githubusercontent.com/59636424/140922347-b26adf3c-1f98-44c1-9621-9aba0e0d2cc5.PNG)

: Text-to-Text Format으로 자연어처리 문제 해결

### 2-2. Find-tuning T5

![ㄷㄷㄷㄷ](https://user-images.githubusercontent.com/59636424/140923279-e15d4472-a829-45c6-b886-029b91a01d9b.PNG)

미리 학습된 pre-trained T5를 활용

-> Fine-tuning: MRC 데이터셋의 QA pair를 활용


## 3. Experiment Results & Analysis

Dataset: Open-domain QA 데이터셋 or MRC 데이터셋에서 지문 제거하고 질문과 답변만 남긴 데이터셋 사용!

Salient Span Masking: 고유 명사, 날짜 등 의미 있는 단위에 속하는 토큰 범위 마스킹한 뒤 학습!

Fine-tuning: T5 checkpoint를 이용해 추가 학습

### 3-1. False negatives

: EM 기준으로 오답으로 채점된 결과를 사람이 평가한 결과 오답이 아닌 경우!

* Pharsing Mismatch: 정답에 대한 표현이 다른 경우
* Incomplete Annotation: 정답이 여러 개일 수 있으나 하나만 정답으로 처리되는 경우
* Unanswerable: 질문을 한 시간이나 문맥에 따라서 정답이 달라지는 경우

### 3-2. Closed QA 한계점

* 모델의 크기가 커서 계산량이 많고 속도가 느림!
* 모델이 어떤 데이터로 답을 내는지 모름
* 모델이 참조하는 지식을 추가하거나 제거하기 어려움!


# 3. 참고 문헌

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

[How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910)

[UnifiedQA: Crossing Format Boundaries With a Single QA System](https://arxiv.org/abs/2005.00700)
