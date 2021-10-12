# 1. Generation-based MRC 배운 것!

## 1. Generation-based MRC

### 1-1. Generation-based MRC 문제 정의

* Extraction-based mrc: context 내 답의 위치 예측 => 분류 문제!
* Generation-based mrc: 주어진 지문과 질의를 보고, 답변을 생성 => 생성 문제!

![gnenr](https://user-images.githubusercontent.com/59636424/136901814-fd35735d-f552-4089-9a9f-7016e9b3ab69.PNG)

-> 정답 위치를 파악하는 것이 아니라 생성하도록 유도!!

### 1-2. Generation-based MRC 평가 방법

EM을 사용하기도 하지만, BLUE도 사용할 수 있다.


### 1-3. Genreation-based MRC Overview

![wwwww](https://user-images.githubusercontent.com/59636424/136902117-ac765305-6653-451b-bb92-87ddd7d1ba93.PNG)

-> extraction-based mrc와 거의 동일하지만 **green box가 score를 예측하지 않고 정답까지 생성한다!**

-> BERT에는 사용 X

### 1-4. Generation-based MRC & Extraction-based MRC 비교

* Seq-to-Seq PLM 구조 (generation) vs PLM + Classifier 구조(extraction)

* text decoding 시, teacher forcing과 같은 방식으로 학습 (generation) vs 지문 내 답의 위치에 대한 확률분포(extraction)

## 2. Pre-processing

extraction based보다 simple!

=> 정답 그대로 넘겨 주면 됨!(정답 index X)

### 2-1. Tokenizer - WordPiece Tokenizer

extraction과 동일

### 2-2. Special Token

Generation MRC는 CLS, SEP 사용할 수 있으나, 정해진 테스트 포맷으로 데이터 생성

![qz](https://user-images.githubusercontent.com/59636424/136903123-72995e9f-d444-49e1-8576-03402da19c33.PNG)

### 2-3. additional information 

* Attention mask는 extraction과 동일
* Token type ids는 BERT와 달리 BART에서는 입력시퀀스 구분이 없어 token_type_ids가 존재 X

=> **generation MRC는 token_type_ids가 들어가지 않는다.**

![rereee](https://user-images.githubusercontent.com/59636424/136903450-947214cc-5c12-4209-ae84-d5e7a9268e06.PNG)

**SEP가 있으면 어느정도 구분이 가능하다!** => question과 context 구분 가능!

### 2-4. 출력 표현 - 정답 출력

generation MRC는 token의 시작과 끝 위치는 필요 없음

![eeeeeeeee](https://user-images.githubusercontent.com/59636424/136903719-fe9872a8-d830-4c10-94bf-a0a1628e4aa9.PNG)

## 3. Model

BERT는 encoder만 존재, BART는 encoder와 decoder 존재

![w](https://user-images.githubusercontent.com/59636424/136904026-37b12c31-a194-472b-899b-447443048038.PNG)

=> pretrain 시에, BERT는 단어 2개 정도를 masking한 후 맞추지만 BART는 기존 문장을 비슷한 방법으로 masking하지만 정답을 생성하는 방식

그래서 BART를 denoising autoencoder라고 한다!!

### 3-1. BART encoder & decoder

* BART encoder -> BERT와 같은 bi-directional
* BART decoder -> GPT와 같은 uni-directional(autoregressive)

### 3-2. Pre-training BART

텍스트 노이즈를 주고 원래 텍스트를 복구하는 문제로 pretrain함!!

![wwwwwww](https://user-images.githubusercontent.com/59636424/136904375-63b9a406-68b6-4425-b2da-19f60d1a1aca.PNG)

## 4. Post-processing

### 4-1. Searching


![searching](https://user-images.githubusercontent.com/59636424/136904496-662632a4-077f-42cf-aab1-d590083d8b3f.PNG)

Exhaustive Search는 모든 가능성을 본다!! => 문장의 길이가 길어지면 불가능!

**Beam Search는 각 timestep마다 top k만 유지!**







