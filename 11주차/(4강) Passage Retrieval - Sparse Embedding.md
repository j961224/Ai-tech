# 1. Passage Retrieval - Sparse Embedding 정리!

## 1. Introduction to Passage Retrieval

### 1-1. Passage Retrieval

질문에 맞는 문서 찾기!

![asas](https://user-images.githubusercontent.com/59636424/136986470-ed0d4227-01c8-4f1a-b433-bad3deee5cf0.PNG)

### 1-2. Passage Retrieval with MRC

* Open-domain Question Answering: 대규모 문서 중에서 질문에 대한 답 찾기

![qqqq](https://user-images.githubusercontent.com/59636424/136986889-6e07afac-ed44-40ea-bf91-a097a0ad0d3f.PNG)

Passage Retrieval로 질문에 대한 답이 있을 것 같은 지문을 찾아 MRC에게 넘기고 MRC는 지문을 읽어 정확한 답변을 읽는 법!

### 1-3. Overview of Passage Retrieval

Query와 Passage Embedding을 한다! -> Passage는 미리 Embedding을 하여 효율성 높임!

각 Passage의 Embedding과 Similarity Score를 측정해 Ranking을 측정한다!

![zzzz](https://user-images.githubusercontent.com/59636424/136987517-5e053789-e9c2-4ca2-95e4-8d3302fda7c1.PNG)

## 2. Passage Embedding and Sparse Embedding

### 2-1. Passage Embedding Space

Passage Embedding의 벡터 공간을 뜻한다.

**문서 간의 유사도, 문서와 질문에 대한 유사도를 벡터 space 상에서 거리로 계산하거나 inner product로 계산 가능하다.**

### 2-2. Sparse Embedding 소개

* Bag-of-Words

각 문서에 존재하는 단어를 1이나 0으로 표현 (있으면 1, 없으면 0)

![rrrrrr](https://user-images.githubusercontent.com/59636424/136988425-debc5d37-8626-409c-93d7-110d2b82b602.PNG)

> * unigram(1-gram) bow
> * bigram(2-gram) or n-gram bow => n-gram까지 가면 기하급수적으로 많아진다.

### 2-3. Sparse Embedding 특징

**Dimension of embedding vector = number of terms**

=> 등장하는 단어가 많아질수록 증가하고 n-gram의 n이 커질수록 증가한다!

=> n이 커질수록 기하급수적으로 vector 크기가 커진다.

**Term overlap을 잡아내는데 유용!**

=> 검색 활용 시, 검색 단어가 실제로 들어가 있는지 본다.

=> **의미가 비슷하지만, 다른 단어의 경우에는 bag of words로는 구별 불가**

## 3. TF-IDF

* TF: 단어 등장 빈도
* IDF: 단어 제공하는 정보의 양

**문서 내에 단어가 많이 등장했지만, 전체 문서에는 잘 안 등장했다면 그 문서에는 중요 단어이므로 점수 더 부과**

![ttt](https://user-images.githubusercontent.com/59636424/136991157-07ba56a8-ca0c-4b80-819d-06bf834c81cf.PNG)

=> best나 times 단어가 IDF 점수가 더 많이 나온다.

### 3-1. TF(Term Frequency)

횟수 count 후 normalize한다!!

### 3-2. IDF

![tttttttt](https://user-images.githubusercontent.com/59636424/136991501-177d3090-2b01-4796-9309-d4c6916cd7e0.PNG)

-> 모든 문서에 등장하는 단어는 IDF score 0점이 된다.

-> 한 문서에만 등장한다면, 큰 수치의 IDF score가 나온다.

### 3-3. Combine TF & IDF

![zzzz](https://user-images.githubusercontent.com/59636424/136991801-626af193-804a-42f8-875e-30a2b9fa77ea.PNG)

### 3-4. TF-IDF 계산 예시

* TF 계산

![ccccc](https://user-images.githubusercontent.com/59636424/136992068-796ada28-a3f7-4132-9323-7c712cebbdbf.PNG)

* IDF 계산

![zzzzzz](https://user-images.githubusercontent.com/59636424/136992206-be268085-1b3d-4e35-99ac-e202ae196447.PNG)

자주 출현한 단어들은 IDF 값이 낮음!

### 3-5. TF-IDF 이용해 유사도 구해보기

TF-IDF는 cosine distance를 사용한다.

### 3-6. BM25

**TF-IDF 개념 + 문서의 길이까지 고려!!**

=> 작은 문서에 더 가중치를 둔다.

![yyyyy](https://user-images.githubusercontent.com/59636424/136992508-1fbc8548-39e1-4adc-a421-2b6180a413fe.PNG)

# 2. TF-IDF를 활용한 Passage Retrieval 실습!

KorQuAD 데이터셋 사용

* 문서만 따로 빼서 사용

~~~
corpus = list(set([example['context'] for example in dataset['train']]))
~~~

* 기본적인 띄워쓰기를 기준으로 token 나누기

~~~
tokenizer_func = lambda x: x.split(' ')
~~~

* TF-IDF 사용(unigram과 bigram 고려)

~~~
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1,2))
~~~

* corpus fit으로 정의 후 transform으로 변형

~~~
vectorizer.fit(corpus)
sp_matrix = vectorizer.transform(corpus)

# sp_matrix[0] -> 첫 번째 문서
~~~

## TF-IDF embedding을 활용하여 passage retrieval 실습해보기

* 질문과 context 가져오기

~~~
query = dataset['train'][sample_idx]['question']
ground_truth = dataset['train'][sample_idx]['context']
~~~

* query tf-idf 나타내기

~~~
query_vec = vectorizer.transform([query])
~~~


* 현재 질문과 각 문서와 유사도 나타내기

~~~
result = query_vec * sp_matrix.T
~~~

=> 변환된 query vector를 document들의 vector과 dot product를 수행

* score 오름차순 하기

~~~
sorted_result = np.argsort(-result.data)
doc_scores = result.data[sorted_result]
doc_ids = result.indices[sorted_result]
~~~

=> result.data는 score를, result.indices는 문서 index 저장





