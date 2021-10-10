# (9강) GPT 언어모델 기반의 자연어 생성 실습!

## (9강) GPT 기반 자연어 생성 모델 학습 - 0_자연어_생성법

KoGPT-2를 사용해보자!

~~~
!apt-get install git-lfs
!git lfs install
!git clone https://huggingface.co/taeminlee/kogpt2
~~~

-> 모델을 직접 다운받기!


~~~
tokenizer = SentencePieceBPETokenizer("/content/kogpt2/vocab.json", "/content/kogpt2/merges.txt") #tokenizer 불러오기

config = GPT2Config(vocab_size=50000) # vocab_size를 맞춰주기
config.pad_token_id = tokenizer.token_to_id('<pad>') #token pad를 명시적으로 알려줌
model = GPT2LMHeadModel(config)

model.load_state_dict(torch.load(model_dir, map_location='cuda'), strict=False) #gpu 사용!
~~~

### Greedy Search

![ㅈㅈㅈㅈㅈㅈ](https://user-images.githubusercontent.com/59636424/136692439-043e17e2-ff79-4cb2-9eb2-1ed87e912dc7.PNG)

확률적으로 높은 것을 선택!

~~~
input_ids = tokenizing("이순신은 조선 중기의 무신이다.")

greedy_output = model.generate(input_ids, max_length=100) # 자동으로 greedy search가 된다!
~~~

=> **하지만 자연스러운 단어가 나오지 않는다!**

### Beam Search

![ㄷㄷㄷ](https://user-images.githubusercontent.com/59636424/136692540-562fd608-39ef-4dfa-96dc-8f9dd2eeef64.PNG)

전체적으로 문장 생성 후, 문장 확률이 최대가 되도록 생성!

=> **문법적으로 잘 맞게 된다!**

=> 한편으로는, inference 시간이 오래 걸린다! => 모두 listup 되므로

~~~
beam_output = model.generate(
    input_ids,  
    max_length=50, 
    num_beams=5, 
    early_stopping=True
)

print(tokenizer.decode(beam_output.tolist()[0], skip_special_tokens=True))

이순신은 조선 중기의 무신이다.</s><s> 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후, 그 후
~~~

=> 반복되는 단어 생김!

=> 이거를 **n-gram(fasttext에서 음절단위로 자르는 행위) 패널티를 이용해서 해결**

=> **n-gram으로 반복되서 나오면 그 쪽길로 안 가게 한다!**

### n-gram 패널티

~~~
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)
~~~

=> **하지만 n-gram 패널티 사용 시, 고유 명사 같은 경우 반복되서 못 나오게 할 수 있으므로 신중히 사용!**

~~~
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    num_return_sequences=5, 
    early_stopping=True
)

0: 이순신은 조선 중기의 무신이다.</s><s> 그 후, 그 해 12월 16일(음력 10월 17일)에 향년 60세를 일기로 사망하였으며, 그의 유해는 서울특별시 서초구 서초동에 있는 국립묘지에 안장되어 있다.(</s>
1: 이순신은 조선 중기의 무신이다.</s><s> 그 후, 그 해 12월 16일(음력 10월 17일)에 향년 60세를 일기로 사망하였으며, 그의 유해는 서울특별시 서초구 서초동에 있는 국립묘지에 안장되었다. .</s>
2: 이순신은 조선 중기의 무신이다.</s><s> 그 후, 그 해 12월 16일(음력 10월 17일)에 향년 60세를 일기로 사망하였으며, 그의 유해는 서울특별시 서초구 서초동에 있는 국립묘지에 안장되어 있다.&
3: 이순신은 조선 중기의 무신이다.</s><s> 그 후, 그 해 12월 16일(음력 10월 17일)에 향년 60세를 일기로 사망하였으며, 그의 유해는 서울특별시 서초구 서초동에 있는 국립묘지에 안장되어 있다. (
4: 이순신은 조선 중기의 무신이다.</s><s> 그 후, 그 해 12월 16일(음력 10월 17일)에 향년 60세를 일기로 사망하였으며, 그의 유해는 서울특별시 서초구 서초동에 있는 국립묘지에 안장되어 있다. -
~~~

=> num_return_sequences는 반환 sequence 갯수를 지정한다.

=> **하지만 유사한 문장만 나오는 것을 확인! -> 오히려 랜덤성, 노이즈를 넣는것이 자연스럽게 나올 것이다!**

### sampling

조건부확률에 따라 단어 무작위로 선정

![ㅂㅈㅂㅈㅂㅈㅂㅈㅂㅈㅂㅈㅂㅈ](https://user-images.githubusercontent.com/59636424/136692835-507698f7-2720-4fb1-86f0-db66950d5383.PNG)

~~~
sample_output = model.generate(
    input_ids, 
    do_sample=True, # 완전 random sampling
    max_length=50, 
    top_k=0 # w/o top_k 추출 -> top_k 0은 전체적으로 랜덤하게 만든다!
)

이순신은 조선 중기의 무신이다.</s><s> 셋업샤크는 또한 은(Ave) 빌리어(Billir) 티아(Tya) 멜릭(Mexelong) 장난감(Billir Music) 피셔
~~~

* temperature 적용!

softmax를 이용한다.

![ㄱㄱㄱ](https://user-images.githubusercontent.com/59636424/136692895-3d7852fb-22c4-43ac-8143-b216e9b20b00.PNG)

~~~
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0, 
    temperature=0.7
)

이순신은 조선 중기의 무신이다.</s><s> 이에 대해 일본 내 시민단체인 일본시민연맹(RSO)은 '한국인을 모욕하는 행위'라고 비난하였다.</s><s> 일본 외무성은 "일본의 정상적인 외교 활동이며 한일 관계를 악화시키는 것을
~~~

### Top-K Sampling

높은 확률 단어 K개를 두고 sampling을 한다!

~~~
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=50
)

이순신은 조선 중기의 무신이다.</s><s> 이 외에도 '노작가의 말' '그저나 거기' 같은 제목과 '기말 고사' 등 다양한 단어를 사용해 사전을 읽어나가기도 했다.</s><s> 당시, 사전에 대한
~~~

### Top-p (nucleus) sampling

k개만 보되, 누적 확률이 특정 확률 이상으로만 넘은거만 본다!!

~~~
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_p=0.92, 
    top_k=0
)

이순신은 조선 중기의 무신이다.</s><s> 스페인은 그 날로 군대 칸세바스에 항의해 군함 전차선 운용을 중지시키고, 빈군에 충의를 바치고, 올 6월 페데리코 칸세바스에 대한 명령으로 전투 장교
~~~

~~~
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

0: 이순신은 조선 중기의 무신이다.</s><s> 이는 약 2억8000만년 전부터 시작된 것으로, 지구 지대의 가장 광범위한 지역과 가장 적은 지역에 걸쳐 서식하였다.</s><s> 가장 많은 종은 라스트와 라센(Bastr)으로, 약
1: 이순신은 조선 중기의 무신이다.</s><s> '도깨비'( )에서 '우연(理神功)'을 언급한 '귀신(龜)'을 통해 이 이야기가 다시 한번 표현되고 있는 셈이다.</s><s> '귀신(龜
2: 이순신은 조선 중기의 무신이다.</s><s> 미국, 독일 및 오스트리아, 벨기에 등 북유럽의 주요 국가들도 이 법에 의해서 설립되었다.
~~~






# 학습 회고!
