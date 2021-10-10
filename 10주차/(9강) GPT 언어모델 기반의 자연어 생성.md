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


## (9강) GPT 기반 자연어 생성 모델 학습 - 1_Few_shot_learning

~~~
sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=512, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1,
        eos_token_id=tokenizer.token_to_id("</s>"), #eos_token_id 지정하면 문장 생성 시, 이것이 등장하면 early_stopping 가능!
        no_repeat_ngram_size=2,
        early_stopping=True
    )
~~~

### Zero-shot learning

~~~
get_gpt_output("<s>철수 : 영희야 안녕!</s><s>영희 : ")

철수 : 영희야 안녕! 영희 : ! - ^^* 아!!
~~~

### Few-shot learning

~~~
get_gpt_output("<s>철수 : 영희야 안녕!</s><s>영희 : 어! 철수야! 오랜만이다!</s><s>철수 : 그러게~ 잘 지냈어?</s><s>영희 : ")

철수 : 영희야 안녕! 영희 : 어! 철수야! 오랜만이다! 철수 : 그러게~ 잘 지냈어? 영희 : 훗, 잘 지내지? - 난 오늘도 오늘도 학교 안 가네 - 수진 : 무슨?!- 영 희진아 뭐 해? : 네, 괜찮아요.
~~~

### 감정 분류(one-shot learning)

~~~
get_gpt_output("<s>본문 : 아.. 기분 진짜 짜증나네ㅡㅡ</s><s>감정 : 분노</s><s>본문 : 와!! 진짜 너무 좋아!!</s><s>감정 : ")

본문 : 아.. 기분 진짜 짜증나네 감정 : 분노 본문 : 와!! 진짜 너무 좋아!! 감정 : ♥.
~~~

### Open Domain Question(Few-shot learning)

~~~
get_gpt_output("<s>질문 : 코로나 바이러스에 걸리면 어떻게 되나요?</s>\
<s>답 : COVID-19 환자는 일반적으로 감염 후 평균 5 ~ 6 일 (평균 잠복기 5 ~ 6 일, 범위 1 ~ 14 일)에 경미한 호흡기 증상 및 발열을 포함한 징후와 증상을 나타냅니다. COVID-19 바이러스에 감염된 대부분의 사람들은 경미한 질병을 앓고 회복됩니다.</s>\
<s>질문 : 코로나 바이러스 질병의 첫 증상은 무엇입니까?</s>\
<s>답 : 이 바이러스는 경미한 질병에서 폐렴에 이르기까지 다양한 증상을 유발할 수 있습니다. 질병의 증상은 발열, 기침, 인후통 및 두통입니다. 심한 경우 호흡 곤란과 사망이 발생할 수 있습니다.</s>\
<s>질문 : 딸기 식물의 수명주기는 무엇입니까?</s>\
<s>답 : 딸기의 생애는 새로운 식물의 설립으로 시작하여 2 ~ 3 년 후 절정에 이르렀다가 절정에 이어 2 ~ 3 년에 노화와 죽음을 향해 진행됩니다. 이상적인 조건에서 딸기 식물은 5-6 년까지 살 수 있습니다.</s>\
<s>질문 : 파이썬 메서드의 self 매개 변수의 목적은 무엇입니까?</s>\
<s>답 : self 매개 변수는 클래스의 현재 인스턴스에 대한 참조이며 클래스에 속한 변수에 액세스하는 데 사용됩니다.</s>\
<s>질문 : 뇌의 어떤 부분이 말을 제어합니까?</s>\
<s>답 : 언어 우세 반구의 왼쪽 전두엽 (브로카 영역)에있는 뇌의 분리 된 부분에 대한 손상은 자발적 언어 및 운동 언어 제어 사용에 상당한 영향을 미치는 것으로 나타났습니다.</s>\
<s>질문 : 인공지능의 미래에 대해 어떻게 생각하십니까?</s>\
<s>답 : ")

[ ~~ 답 : rain_NVS가 현재 개발 중인 urgical music control quant cruisine에 의해 연구된다면 인공지능이 과연 인공지능에게 얼마나 중요한 의미를 가지는 지를 보여줌니다. money에 대한 자신의 생각을 말하기 위해서입니다.]
~~~

=> 질문, 답 형태로 input을 넣음!

### 번역

~~~
get_gpt_output("<s>한국어: 그 도로는 강과 평행으로 뻗어 있다.</s>\
<s>English: The road runs parallel to the river.</s>\
<s>한국어: 그 평행선들은 분기하는 것처럼 보인다.</s>\
<s>English: The parallel lines appear to diverge.</s>\
<s>한국어: 그 도로와 운하는 서로 평행하다.</s>\
<s>English: The road and the canal are parallel to each other.</s>\
<s>한국어: 평행한 은하계라는 개념은 이해하기가 힘들다.</s>\
<s>English: The idea of a parallel universe is hard to grasp.</s>\
<s>한국어: 이러한 전통은 우리 문화에서는 그에 상응하는 것이 없다.</s>\
<s>English: This tradition has no parallel in our culture.</s>\
<s>한국어: 이것은 현대에 들어서는 그 유례를 찾기 힘든 업적이다.</s>\
<s>English: This is an achievement without parallel in modern times.</s>\
<s>한국어: 그들의 경험과 우리 경험 사이에서 유사점을 찾는 것이 가능하다.</s>\
<s>English: It is possible to draw a parallel between their experience and ours.</s>\
<s>한국어: 그 새 학위 과정과 기존의 수료 과정이 동시에 운영될 수도 있을 것이다.</s>\
<s>English: The new degree and the existing certificate courses would run in parallel.</s>\
<s>한국어: 이순신은 조선 중기의 무신이다.</s>\
<s>Englisth: ")
~~~

## (9강) GPT 기반 자연어 생성 모델 학습 - 2_KoGPT_2_기반의_챗봇 (한국어 언어모델 학습 및 다중 과제 튜닝)

~~~
tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>") # padding token 넣기 -> padding 설정
tokenizer.enable_truncation(max_length=128) # truncation max length 알려주기
~~~

* 실습 class ChatDataset 중, load_data 함수

~~~
def load_data(self):
        raw_data = pd.read_csv(self.file_path)
        train_data = '<s>'+raw_data['Q']+'</s>'+'<s>'+raw_data['A']+'</s>'
        #<s>안녕하세요</s><s> -> 네, 안녕하세요</s>
~~~

-> one-shot, few-shot은 생성 시, 앞부분에 정보를 줌

-> **fine-tuning은 좀 더 생성 패턴 자체를 이렇게 만들도록 함!**

* 학습 중 코드

~~~
outputs = model(data, labels=data)
~~~

=> labels이 data 자체이므로, data input을 1칸씩 넣어가면서 다음 token이 뭐가 나오는지 확률 최대가 되도록 학습!

~~~
sample_outputs = model.generate(
        input_ids,
        num_return_sequences=5,
        do_sample=True, 
        max_length=128, 
        top_k=50, 
        top_p=0.95, 
        eos_token_id=e_s,
        early_stopping=True,
        bad_words_ids=[[unk]] #unk token 등장 시, 다른 것을 선택하도록 명시!
    )
~~~



# 학습 회고!
