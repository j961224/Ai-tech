# 1. Linking MRC and Retrieval 정리!

## 1. ODQA introduction

### 1-1. Linking MRC and Retrieval: ODQA

MRC-> 지문이 주어진 상황에서 질의응답을 하는 것!

ODQA -> **지문이 특정해서 주어지는 것이 아니라 web 전체가 주어짐!**

![ttttt](https://user-images.githubusercontent.com/59636424/137093965-ca4d5d2c-d1c5-4ff3-80b5-5bb05123d22b.PNG)

**따라서 봐야할 문서가 매우 많다!**

### 1-2. ODQA 역사

* Text retrieval conference (TREC) 

연관 문서만 반환하는 information retrieval을 넘어서서, **short answer 형태가 목표**이다!

3단계로 나뉨(Question processing + Passage retrieval + Answering processing)

### 1-3. Question processing

질문으로부터 키워드 선택! -> Answer type 선택

ex) 질문은 답변 형태가 '장소'여야 하는 것을 미리 선택

### 1-4. Passage retrieval

연관된 문서 뽑기 -> passage 단위로 자름(Named entity/ Passage 내 question 단어 개수 등을 활용) -> 그 후, 선별

현재: TF-IDF나 BM25로 많이 사용(문서 뽑기)

### 1-5. Answer processing

feature들을 활용해서 classifier 만들기 -> 주어진 question에서 passage 선택!

현재: passage 뿐만 아니라, span까지 선택!

### 1-6. IBM Watson(2011)

## 2. Retriever-Reader 접근법

### 2-1. Retriever-Reader 접근 방식

* Retriever: DB에 문서 찾기
* Reader: Retriever가 찾은 문서로 질문에 대한 답 찾기!


|type|Retriever|Reader|
|---|----|-----|
|입력|문서셋, 질문|Retriever를 통한 문서, 질문|
|출력|관련성 높은 문서|답변|
|학습단계|TF-IDF나 BM25는 없지만 Dense는 학습!|MRC 데이터셋으로 학습|

* Dense는 QA data로 학습!

* Reader는 Distance supervision으로도 학습한다!

### 2-2. Distant supervision

질문-답변만 있는 데이터셋(CuratedTREC) -> 답이 어느 지문에 있는지는 모름

그래서 **답의 위치를 찾아야한다! -> 이것이 Distant supervision!**

    1. Retriever로 관련성 높은 문서를 검색!
    
    2. 짧거나 고유 명사가 없는 문서 제거
    
    3. answer가 exact match가 없는 문저 제거
    
    4. 질문과 연관성이 가장 높은 단락 -> 답의 위치로 선정!
    
![wwwwww](https://user-images.githubusercontent.com/59636424/137097945-c16ccf72-f312-4141-a607-da49947115a3.PNG)

위는 적용 예시!

### 2-3. inference

Retriever가 질문과 **관련 높은 5개 문서 선택** -> Reader는 5개 **문서 읽고 답변 예측** -> 예측 답변 중 **가장 score가 높은 것이 최종 답!**

## 3. Issues & Recent Approaches

### 3-1. passage 단위가 엄밀히 정의되지 않았다.

passage를 문서 단위, 단락, 문장 단위로 볼 수 있다.

### 3-2. granularities(세분성)

Retriever 단계에서 몇개 문서를 넘길지 결정!

각 문서, 단락, 문장 단위마다 넘기는 k가 다를 수 밖에 없다!

### 3-3. Single-passage training vs Multi-passage training

* 학습 시, single-passage 만 본다면?

k개의 passage -> reader가 확인 -> 특정 answer span에 대한 예측 점수 측정 -> 그 중, 가장 높은 점수를 가진 answer span 선택!

각 retrieved passage들의 직접적 비교 X -> 비교 자체가 불가

* 학습 시, Multi-passage는?

retrieved passage 전체 -> 하나의 apssage로 취급 -> reader 모델은 answer span 하나만 찾도록 함

더 많은 GPU와 메모리 할당!!

### 3-4. each passage의 중요성

retriever 모델에서 추출된 top-k passage들의 score를 그대로 reader에게 전달한다면?

=> 최종 answer ranking 시, retriever passage score를 받아서 사용하면 더 좋은 효과가 날 수 있다.

![wewewewe](https://user-images.githubusercontent.com/59636424/137100860-8dc6bef3-a42e-4beb-8090-758451ee9b42.PNG)

# 2. Open-Domain Question Answering(ODQA) 시스템 구축 실습 정리!

### context tfidf를 통해 sparse retrieve 만들기

```python
vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1,2))
sp_matrix = vectorizer.fit_transform(corpus)
```

### 질문에 따른 score 높은 순으로 top k개 뽑기

```python
def get_relevant_doc(vectorizer, query, k=1):
    # top k
    """
    참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    """
    query_vec = vectorizer.transform([query]) #query score화
    # sum이 0이면 search 불가
    assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
    
    result = query_vec * sp_matrix.T # query와 context score 계산
    sorted_result = np.argsort(-result.data) # 내림차순으로 정렬
    doc_scores = result.data[sorted_result] #높은 순으로 score 저장
    doc_ids = result.indices[sorted_result] #높은 score 순으로 context index 저장
    return doc_scores[:k], doc_ids[:k] # top k개만큼 뽑기
```

### 질문에 대한 답변을 context에 찾기

```python
def get_answer_from_context(context, question, model, tokenizer):
    #encoder_dict: tokenizer로 전부 encoding해서 bert에 넣어주기
    encoded_dict = tokenizer.encode_plus(  
        question,
        context,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    #padding 해주기
    non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
    
    # non_padded_ids를 decode -> padding과 CLS token 같은 거 없애주기
    full_text = tokenizer.decode(non_padded_ids)
    
    
    inputs = {
    'input_ids': torch.tensor([encoded_dict['input_ids']], dtype=torch.long),
    'attention_mask': torch.tensor([encoded_dict['attention_mask']], dtype=torch.long),
    'token_type_ids': torch.tensor([encoded_dict['token_type_ids']], dtype=torch.long)
    }
    
    outputs = model(**inputs)
    start, end = torch.max(outputs.start_logits, axis=1).indices.item(), torch.max(outputs.end_logits, axis=1).indices.item()
    answer = tokenizer.decode(encoded_dict['input_ids'][start:end+1])
    return answer
```

-> model을 가지고 start와 end 부분을 가져와서 decoding!

### 통합해서 ODQA 시스템 구축

```python
def open_domain_qa(query, corpus, vectorizer, model, tokenizer, k=1):
    # 1. Retrieve k relevant docs by usign sparse matrix
    _, doc_id = get_relevant_doc(vectorizer, query, k=1)
    context = corpus[doc_id.item()]

    # 2. Predict answer from given doc by using MRC model
    answer = get_answer_from_context(context, query, mrc_model, tokenizer)
    print("{} {} {}".format('*'*20, 'Result','*'*20))
    print("[Search query]\n", query, "\n")
    print(f"[Relevant Doc ID(Top 1 passage)]: {doc_id.item()}")
    pprint(corpus[doc_id.item()], compact=True)
    print(f"[Answer Prediction from the model]: {answer}")
```

-> 앞서 구한, get_relevant_doc, get_answer_from_context로 query와 제일 연관 있는 docx와 그 docx로 span answer를 구하는 과정을 통합한 것이다!
