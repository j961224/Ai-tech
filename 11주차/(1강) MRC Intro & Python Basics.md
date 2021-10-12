# 1. (1강) MRC Intro & Python Basics 정리!


## 1. Introduction to MRC

### MRC 개념

기계 독해로 주어진 지문을 이해하고 질의를 답변을 추론하는 문제이다!

### MRC 종류

#### 1. Extractive Answer Datasets

질의에 대한 답이 항상 주어진 지문(context)의 segment로 존재!!

=> **Span Extraction이라는 것은 위에서 말했듯이 지문에서 답이 존재하여 추출**하는 것이다!

ex) SQuAD, KorQuAD, NewsQA 등

#### 2. Descripitve/Marrative Answer Datasets

답이 지문 내에 추출하는 것이 아닌 **생성된 sentence의 형태!**

ex) MS MARCO, Narrative QA

#### 3. Multiple-chooice Datasets

질의에 대한 답을 여러 개의 answer 후보 중 하나로 고르는 형태!

ex) MCTEst, RACE, ARC 등

#### 4. MRC 어려운 점

* **단어들의 구성이 유사하지는 않지만 동일한 의미의 문장을 이해!!**

ex) DuoRC/QuoRef

> * Paraphrasing: 같은 의미를 가졌지만 다른 단어를 사용
> * Coreference Resolution: 그것, 그 사람과 같은 지칭하는 용어가 실제로 무엇을 지칭하는지를 알아야 지문 이해가 가능하므로 지칭하는 것을 찾아내는 task

* **질문 내에 답변이 존재하지 않는 경우**

이 경우, 'No Answer'이라고 말하는게 자연스럽다!

ex) SQuAD 2.0

* **Multi-hop reasoning**

여러 문서에서 질의에 대한 fact를 찾아 답을 찾아내는 것!

ex) HotpotQA, QAngaroo

#### 5. MRC 평가 방법

* Exact Match/F1 Score: 지문 중에 답변 추출, 다중 선택 답변 dataset 평가에 주로 쓰인다!

**EM or Accuracy**: 예측 답과 ground-truth가 정확히 일치하는 샘플의 비율

**F1 score**: 예측한 답과 ground-truth 사이의 overlap을 준다. -> 약간 틀릴 경우에 어느 정도 스코어 부여

**descriptive answer에는 쓰이지 못 하는 완벽하게 일치하는 경우가 거의 없으므로 위의 평가 지표를 쓸 수 없다!**

---

* ROUGE-L/BLEU: 단어 뿐만 아니라, n-gram 등 여러 개 단어가 겹치는지를 본다.

**ROUGE_L Score**: overlap recall

**BLUE**: precision

## 2. Unicode & Tokenization

### 1. Unicode란?

문자를 일관되게 표현하게 다룰 수 있도록 한다!

### 2. 인코딩 & UTF-8

**인코딩**: 문자를 컴퓨터에서 저장 및 처리할 수 있게 이진수로 바꾸는 것!

**UTF-8**: UTF-8는 현재 가장 많이 쓰는 인코딩 방식!!

### 3. Python에서 Unicode 다루기!

Python3부터 string 타입 -> 유니코드 표준 사용!

* ord: 문자 -> 유니코드 code point
* chr: code point -> 문자

---

**Unicode와 한국어는 한자 다음으로 유니코드에서 많은 코드를 차지하고 있다.**

* 완성형: 모든 완성형 한글 11172자!
* 조합형: 초성, 중성, 종성을 조합하여 만든 글자

### 4. 토크나이징

단어(띄어쓰기 기준), 형태소, subword 등 여러 토큰 기준으로 사용!

### 5. subword 토크나이징

~~~
sentence='아버지 가방에 들어가신다'

tokenizer.tokenize('아버지 가방에 들어가신다')
['아버지', '가', '##방', '##에', '들어', '##가', '##신', '##다']
~~~

**띄어쓰기 기준으로 한다면, '아버지', '가방에', '들어가신다'로 나뉘는데 단어가 너무 커져 비교가 어려움!!**

**subword로 나눌 시, 자주 나오는 단어있는지를 보고 자른다.** -> ##은 한 단어가 2개 이상으로 나뉠 시, ##이 들어간다.

### 6. BPE

**자주 나오는 character sequence를 하나의 단어로 인식하고자 한다.**

![qqq](https://user-images.githubusercontent.com/59636424/136885246-a3f916c2-a29f-4f80-bc5a-5678d8dce5fa.PNG)

## 3. Dataset 살펴보기

### KorQuAD

LG CNS가 개발한 데이터셋!

-> 영어에서 쓰이는 모델을 한국어에도 쓰일 수 있도록 되었다!

![ww](https://user-images.githubusercontent.com/59636424/136885580-ba273e40-e172-4f66-a1dc-9715d8d6fd33.PNG)

1550개의 위키피디아 문서에 대해서 63952개 질의응답 쌍으로 만들었다!

### KorQuAD 데이터 수집 과정

대상 문서 수집(짧은 문단, 수식이 포함된 문단 제거) -> 질문/답변 생성(질의응답 쌍을 생성) -> 2차 답변 태깅

### HuggingFace datasets 라이브러리 소개

HuggingFace datasets 접근방법!

~~~
from datasets import load_dataset
#squad_kor_v1, squad_kor_v2
dataset = load_dataset('squad_kor_v1', split='train')
~~~

다른 데이터셋에 쉽게 적용 가능!

메모리 공간 부족 및 전처리 과정 반복의 번거로움 제거!!!

### KorQuAD 예시

![eeee](https://user-images.githubusercontent.com/59636424/136885963-23645d79-ac19-452d-be3d-0db786cd5aad.PNG)

### KorQuAD 답변 유형

![zs](https://user-images.githubusercontent.com/59636424/136886438-4c7c1b92-da34-4a51-b61b-c28e7c159ef4.PNG)

대상, 인물, 시간, 장소, 방법, 원인


# 2. 실습 HuggingFace 빠르게 훑기!

## 1. Huggingface Transformer 훑기

### tokenizer

* tokenizer 가져오기

~~~
tokenizer = AutoTokenizer.from_pretrained(model_name)
~~~

* tokenizer 사용 시, train data의 언어를 이해할 수 있는 tokenizer인지 확인!!
* 사용하고자 하는 pretrained model과 동일한 tokenizer인지 확인!

### Config

* 모델과 동일한 config 가져오기

~~~
model_config = AutoConfig.from_pretrained(model_name)
~~~

* hidden dim 등은 정해져 있으니 바꾸면 안 된다!
* special token을 추가 시, vocab size도 추가해서 학습해야한다!

~~~
model_config = AutoConfig.from_pretrained(model_name)
model_config.vocab_size = model_config.vocab_size + 2
~~~

### Pretrain model 불러오기

> * 기본 모델: hidden state가 출력되는 기본 모델
> * downstream task 모델: 기본 모델 + head가 설정된 모델

* config와 pretrained()로 불러오기

~~~
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
        model_name, config=model_config
    )
~~~

## 2. Huggingface Trainer

* TrainingArguments 설정
* Trainer 호출
* 학습 / 추론 진행


## 3. Advanced tutorial

### Token 추가하기

* special token 추가하기

~~~
special_tokens_dict = {'additional_special_tokens': ['[special1]','[special2]','[special3]','[special4]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
~~~

* token 추가하기

~~~
new_tokens = ['COVID', 'hospitalization']
num_added_toks = tokenizer.add_tokens(new_tokens)
~~~

* tokenizer config와 model의 token embedding 사이즈 수정!

~~~
config.vocab_size = len(tokenizer)
model.resize_token_embeddings(len(tokenizer))
~~~

-> 만일, 모델에 dummy vocab을 가지고 있다면, resize할 필요가 없다!

### [CLS] output 추출!

* [CLS] 추출하기

~~~
inputs = tokenizer("정유석은 MRC를 참가했다 !!", return_tensors="pt")
outputs = model(**inputs)

cls_output = outputs.pooler_output
~~~

-> .pooler_output을 이용해 cls_output을 추출할 수 있다.

* [CLS] token은 문장을 대표하는가?

**[CLS]가 sentence를 대표하지 못 하는 경우가 있으므로 input의 representation을 추출하여 pooling layer -> maxpooling or average pooling을 시행하기도 함!**

