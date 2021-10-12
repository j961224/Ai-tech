# 1. Extraction-based MRC 정리!


## 1. Extraction-based MRC

### 1-1. Extraction-based MRC 문제 정의

질문 답변이 항상 지문에 존재!!

ex) SQuAD, KorQuAD, NewsQA 등

### 1-2. Extraction-based MRC 평가 방법

EM, F1 Score로 평가!

### 1-3. Extraction-based MRC Overview

![wwwww](https://user-images.githubusercontent.com/59636424/136890351-75e1807e-88ec-4546-8e20-6643ec5148ea.PNG)

    1. Context와 Question을 단어를 쪼개제서 input으로 들어간다.
    
    2. Word Embedding으로 벡터화 시킨 후, 시작점과 끝점을 내보낸다. -> context와 question에 해당하는 contextalize vector를 보낸다!
    
    3. 최종 예측값은 시작과 끝 사이에 있는 span을 그대로 답으로 보낸다!
    
## 2. Pre-processing

* 입력 예시

![wwq](https://user-images.githubusercontent.com/59636424/136890724-86107705-4acb-4597-9df8-06b3125ce24b.PNG)

### 2-1. Tokenization

띄어쓰기 기준. 형태소, subword 등 여러 단위 토큰 기준 사용!

**최근에는 OOV를 해결하기 위한 BPE 사용!** -> 그 중 하나로 WordPiece Tokenizer 사용!

### 2-2. Special Tokens

![qqqqqqqq](https://user-images.githubusercontent.com/59636424/136891088-b4a85b63-25ca-4982-a25f-5882fa13d2f1.PNG)

![rrrrrr](https://user-images.githubusercontent.com/59636424/136891211-15401e22-4d50-4b4e-a482-e99a9bc46cea.PNG)

[CLS] 질문 [SEP] context [SEP] [PAD] x n

### 2-3. Attention Mask

**attention 연산 시, 무시할 토큰을 표시한다!** (0은 무시, 1은 연산 포함) -> [PAD]가 보통 무시된다.

![qwa](https://user-images.githubusercontent.com/59636424/136891418-f55da3b7-5668-4cbc-825c-b6b44f61aa5f.PNG)

### 2-4. Token Type IDs

**질문과 지문을 구분할 수 있도록 각각에게 ID를 부여한다!** -> [PAD]는 보통 0으로 준다.

![zx](https://user-images.githubusercontent.com/59636424/136891559-093da528-9a20-4f72-8f74-7f08e90a5ea3.PNG)

### 2-5. 모델 출력값

정답은 chracter 위치로 파악한다. -> **하지만 tokenize한 다음은, 학습 시 signal을 정답 token이 어디 있는가로 판단!**

**span의 시작 위치와 끝위치를 예측하도록 학습**

## 3. Fine-tuning

### 3-1. Fine-tuning BERT

![ft](https://user-images.githubusercontent.com/59636424/136892050-54b52cfc-45af-475d-981c-0cf416d93a24.PNG)

지문 내에서 정답에 해당되는 embedding을 linear transformation을 통해서 각 단어마다 하나의 숫자로 나올 수 있도록 한다.

## 4. Post-processing

### 4-1. 불가능한 답 제거하기

candidate list에서 제거!

**End position이 start position 보다 앞에 있는 경우 제거!**

**예측한 위치가 context를 벗어난 경우 제거!** (question 위치 쪽에 답이 나오는 경우!)

**설정한 max_answer_length보다 긴 경우!**

### 4-2. 최적의 답안 찾기

    1. start/end position prediction에서 score(logits)가 가장 높은 N개 찾기
    
    2. 불가능한 start/end 조합 제거
    
    3. 가능한 조합들을 score 합 큰 순서대로 정렬
    
    4. Score가 가장 큰 조합 최종 예측
    
    5. Top-k가 필요한 경우 차례대로 내보낸다!
    


# 2. BERT를 활용해서 Extraction-based MRC 문제 풀기 실습!!

~~~
!git clone https://github.com/huggingface/transformers.git
import sys
sys.path.append('transformers/examples/question-answering')
~~~

=> git clone 후, cache에 쓸 수 있도록 한다.

~~~
from datasets import load_dataset

datasets = load_dataset("transformers/squad_kor_v1")
~~~

=> dataset load

~~~
datasets #하나의 테이블로 볼 수 있다.

DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 60407
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 5774
    })
})
~~~

~~~
from dataset import load_metric

metric = load_metric('squad')
~~~

=> squad에 맞는 평가 metric 불러오기


## Pre-trained 모델 불러오기

~~~
from transformers import (
    AutoConfig, # config 불러오는 class 불러오기
    AutoModelForQuestionAnswering, #model 불러오는 class 불러오기
    AutoTokenizer
)
~~~

## 설정하기

~~~
max_seq_length = 384 # 질문과 컨텍스트, special token을 합한 문자열의 최대 길이
pad_to_max_length = True
doc_stride = 128 # 컨텍스트가 너무 길어서 나눴을 때 오버랩되는 시퀀스 길이
max_train_samples = 16
max_val_samples = 16
preprocessing_num_workers = 4
batch_size = 4
num_train_epochs = 2
n_best_size = 20
max_answer_length = 30
~~~

> * pad_to_max_length로 padding한다!
> * doc_stride: 문서가 긴 경우, 일부 겹치게 해서 나누는 방식!
> > 문서 길이가 500이라면(max_seq_length에 안 들어가므로), 2개로 쪼개고 overlap 되는게 128개로 지정 -> 각 문서에 답을 구하고 답을 취합한다. -> 두 개중, 더 확률 높은 거를 가져간다.
> * max_train_samples: 갯수를 적당히 정해서 train를 빨리 끝내게 한다.

## 전처리하기!


~~~
train_dataset.select(range(max_train_samples))
~~~

-> 이 코드로 max_train_samples 수만큼만 data 가져온다.

~~~
train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
        )
~~~

-> mapping을 통해, 전처리 및 효율적으로 데이터 사용시키기!

~~~
eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
        )
~~~

-> 평가 데이터 또한, mapping을 통해 전처리 시키기!

## Fine-tuning하기

~~~
from transformers import default_data_collator # 학습 시, 다른 example들을 collator해준다!
from transformers import TrainingArguments, EvalPrediction
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions #답변 낼 시, 1번 더 postprocess하기!
~~~

~~~
training_args = TrainingArguments(
    output_dir="outputs",
    do_train=True, 
    do_eval=True, 
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
)

trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    
train_result = trainer.train()
~~~

## 평가하기

~~~
metrics = trainer.evaluate()
~~~


