# 1. (8강)GPT언어모델 배운 것!

## 08. GPT 언어 모델

GPT는 자연어 생성 특화 모델! -> **Transformer의 Decoder를 사용**!

![얍얍](https://user-images.githubusercontent.com/59636424/136689530-8aa19561-4f15-4c81-a1d3-75fbd4c4d5ba.PNG)

위의 사진과 같이 '발' token 다음 '없는'을 예측!

![ㅇㅂㅇㅂ](https://user-images.githubusercontent.com/59636424/136689616-1fecef71-db35-41fc-bb0c-0662dfedfed7.PNG)

**GPT-1의 목적은 기존 문장입력 -> 입력된 context vector 출력 -> 그 뒤에 linear layer를 붙여 분류!**

### GPT 모델 소개

#### 장점

GPT1은 **자연어 문장을 분류하는데 좋은 성능**을 낸다!

-> pretrain모델을 사용해 fine-tuning으로 적은 양의 데이터로 좋은 성능!!

#### 단점

**지도 학습이 필요하며 labeled data가 필수!!**

**특정 task에 fine-tuning된 모델은 다른 task에 사용 불가!!**

---

지도학습의 목적학습과 비지도 학습의 목적함수가 같다!(언어 특성상!)

=> 지도학습의 fine-tuning에서 만들어지는 목적함수와 비지도학습의 pretrain의 목적함수가 같다!

=> **fine-tuning에서 사용되는 label 자체도 언어이다!** ex) 기쁨, 슬픔 자체가 언어이다. (감정 분석에서)

![ㅇㅋㅇㅋ](https://user-images.githubusercontent.com/59636424/136689794-de37f886-5fef-4892-afd7-9675c1970686.PNG)

**위와 같이 엄청 큰 데이터셋을 통해 '놀란'이라는 의미를 명확히 말하지 않아도 감정도 자연스럽게 학습이 가능!!**

**그러므로 비지도 학습과 지도 학습의 목적함수를 구분할 필요가 없다!!**


---

![zero](https://user-images.githubusercontent.com/59636424/136690143-0e77c331-adc2-4556-94ee-17f1e8814cd0.PNG)

![few](https://user-images.githubusercontent.com/59636424/136690145-f24eb9d6-2f0b-4a60-a2ec-610ed526b8c6.PNG)

pretrain 모델을 만드는데 많은 자원이 필요한데 fine-tuning으로 1가지 task에만 적용한 모델을 배포하는 것이 낭비!!

=> 그래서 **zero-shot, one-shot 등을 제안!!**

**zero-shot, few-shot, one-shot learning들은 gradient 업데이트가 존재하지 않는다!**

=> inference를 할 때, 원하는 task의 hint의 갯수에 따라 있다!

=> **이 아이디어로 GPT-2를 개발!**

### GPT-2

![rnwh](https://user-images.githubusercontent.com/59636424/136690221-9f5176a9-efd8-49bb-a7af-0f86772c6c91.PNG)

GPT-1의 decoder에서 구조만 좀 다르게 구성!

**다음 단어 예측 방식에서 SOTA 성능!**

**Zero, One, Few-shot learning의 새 지평 제시!**

### GPT-3

기존 GPT-2의 hyper parameter를 엄청 늘린다!

모델 사이즈와 학습 데이터를 키웠다!

GPT-2와는 다른 구조를 사용하고 initalize를 살짝 고친 모델로 되어 있다!

**제목, 부제목 입력 시, GPT-3가 뉴스 기사 생성하고 진짜와 비교할 시 꽤 많은 기사가 비슷했음!**

#### zero-shot, few-shot 실험

수학적 계산에 의한 task 수행

#### Open Domain QA 실험

문서 없이 바로 질문한 실험!

![얍](https://user-images.githubusercontent.com/59636424/136690707-14a86673-1cbf-46e5-9759-61b3a9faffe7.PNG)

바로 SOTA 성능이 나옴!

#### 텍스트 데이터 파싱도 가능!

요약한 자료를 표로도 만들 수 있다!

---

**GPT-3의 문제는 weight update가 불가능하므로 새로운 지식 학습이 불가능하다!**

=> 시기가 달라지는 문제에 대응 불가!! (ex) 현재 대통령은?)

**멀티 모달 정보가 필요!** -> 글로만 배우는 것이 한계이므로 멀티 모달 정보 학습도 필요!


# 2. 실습!

## (8강) GPT 언어 모델 소개 - 0_한국어_GPT_2_pre_training

SentencePiceBPETokenizer를 이용한다!

~~~
from tokenizers import SentencePieceBPETokenizer
from tokenizers.normalizers import BertNormalizer

tokenizer = SentencePieceBPETokenizer()

tokenizer._tokenizer.normalizer = BertNormalizer(clean_text=True,
  handle_chinese_chars=False,
  lowercase=False)
~~~

=> BPE과 내부적으로 동일하지만 underbar를 추가한 tokenizer를 사용!

~~~
tokenizer.train(
    path,
    vocab_size=10000,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
    ],
)
~~~

**BERT와 다른 점은 학습 시, CLS, SEP token을 붙이는데 GPT는 CLS, SEP는 필요없다!**

=> fasettext를 생각할 시, 음절 시작과 끝을 표시하는 special token을 사용하는데 GPT도 마찬가지다!

=> 이것을 이용해서 **문장단위로 생성하기 위해서 <s>, </s> token 추가!**

~~~
print(tokenizer.encode("이순신은 조선 중기의 무신이다.").tokens)

['▁이', '순', '신은', '▁조선', '▁중', '기의', '▁무', '신', '이다.']
~~~

**문장 어절의 시작 부분에는 다 underbar가 붙는다!**

~~~
tokenizer.save_model(".")
~~~~

=> 나만의 SentecePieceBPE tokenizer 완성!

~~~
tokenizer = SentencePieceBPETokenizer.from_file(vocab_filename="vocab.json", merges_filename="merges.txt")
~~~

=> 저장한 tokenizer 불러오기!

~~~
print(tokenizer.encode("<s>이순신은 조선 중기의 무신이다.</s>").tokens)

['▁<', 's', '>', '이', '순', '신은', '▁조선', '▁중', '기의', '▁무', '신', '이다.', '<', '/s', '>']

tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>", "<shkim>"])
tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")
tokenizer.unk_token_id = tokenizer.token_to_id("<unk>")
tokenizer.bos_token_id = tokenizer.token_to_id("<bos>")
tokenizer.eos_token_id = tokenizer.token_to_id("<eos>")

print(tokenizer.encode("<s>이순신은 조선 중기의 무신이다.</s>").tokens)

['<s>', '▁이', '순', '신은', '▁조선', '▁중', '기의', '▁무', '신', '이다.', '</s>']
~~~

=> 아직 special token으로 보지 않으니 add_special_tokens로 token 추가해준다!

### GPT-2 학습하기!

~~~
from transformers import GPT2Config, GPT2LMHeadModel
# creating the configurations from which the model can be made
config = GPT2Config(
  vocab_size=tokenizer.get_vocab_size(),
  bos_token_id=tokenizer.token_to_id("<s>"),
  eos_token_id=tokenizer.token_to_id("</s>"),
)
# creating the model
model = GPT2LMHeadModel(config)
~~~

GPT2Config로 GPT2 껍데기를 만든다!

=> tokenizer.get_vocab_size()로 명확하게 vocab_size 지정!

=> bos_token과 eos_token으로 각 token 지정!

~~~
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(    # GPT는 생성모델이기 때문에 [MASK] 가 필요 없습니다 :-)
    tokenizer=tokenizer, mlm=False,
)
~~~

Transformer의 DataCollatorForLanguageModeling로 lm call 하기!, GPT는 masking이 필요 없으니 mlm=False로 둠!

~~~
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3)
~~~

=> generate로 입력값이 input_ids(tokenize된 vocab id), do_sample(sample 반환해라!),  num_return_sequences(문장 몇개)를 넣음!


# 3. 학습 회고!

GPT는 BERT보다 좀 더 복잡하지 않고 단조로운 것 같고 얼른 transformer를 코드로 구현하면 GPT도 raw로 code를 구현하고 싶다는 생각이 든다.

또한 GPT부터 GPT-3까지 논문을 보고 더욱 꼼꼼히 이해해야겠다!

[GPT논문](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

[GPT-2논문](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[GPT-3논문](https://arxiv.org/abs/2005.14165)

[언어 모델을 가지고 트럼프 봇 만들기!](https://jiho-ml.com/weekly-nlp-19/)


