# 1. Scaling up with FAISS 정리!

## 1. Passage Retrieval and Similarity Search

### 1-1. MIPS(Maximum Inner Product Search)

2개의 벡터를 inner product를 가지고 가장 큰 것을 찾는다.

![xx](https://user-images.githubusercontent.com/59636424/137064968-6e11398d-28b2-43ab-a6a7-6b90c85b09d1.PNG)

=> **가장 score를 많이 내는 i번째 벡터를 찾는다!**

**하지만, 방대한 양일수록, bruth force 방법이니 비효율적이다.**

### 1-2. MIPS & Challenges

실제 검색해야할 데이터는 방대함! => 일일이 보면서 검색할 수 없다!

### 1-3. Tradeoffs of similarity search

![qw](https://user-images.githubusercontent.com/59636424/137065240-79eb66f2-5f92-4b5e-81a8-5b1e9bea35d2.PNG)

=> 문제를 접근하는 방법은 각각 1개씩 존재!

## 2. Approximating Similarity Search

### 2-1. 숫자 Compression 방법(Scalar Quantization)

inner product 시, 4 byte까지 필요로 하지 않는다. => 그래서 1 byte로 압축한다.(Scalar quantization)

![zxzxzxzx](https://user-images.githubusercontent.com/59636424/137065665-5769a12d-3efe-4982-8a11-464f5257d165.PNG)

### 2-2. Pruning- Inverted File

* clustering

점들을 정해진 cluster로 소속시켜서 군집을 이루게 한다! => query가 들어왔을 때, query에 가장 근접한 cluster만 보기!!

전체 vector space를 k개의 cluster로 나눔 => k-means clustering! (자주 사용되는 기법)

![ee](https://user-images.githubusercontent.com/59636424/137065879-b963a62a-9754-45fa-b22c-a5bfd24e71dc.PNG)

* Inverted file(IVF)

각 cluster에 속해있는 point들을 역으로 index를 가지므로 Inverted list structure!

**각 cluster centriod id - 해당 cluster vector들로 연결!**

![rt](https://user-images.githubusercontent.com/59636424/137066109-5b914837-c71c-4b7a-9315-8df9731a11e3.PNG)

## 3. Introduction to FAISS

### 3-1. FAISS

large scale에 특화!!

**indexing 쪽을 효율적으로 도와준다!** -> encoding X

### 3-2. Passage Retrieval with FAISS

* 1. map vector과 index 학습

![weq](https://user-images.githubusercontent.com/59636424/137066730-76527235-7c27-4f2e-a805-028ab6261f0d.PNG)

**학습을 하는 이유는 FAISS활용 시, cluster를 확보해야하는데 cluster들은 데이터 분포를 보고 적절한 cluster를 지정하기 위해서!**

=> **scalar quantization 하는데도 float number max, min에 대해서 파악해야하므로 학습 필요!**

이렇게 구한, cluster랑 sq8이 정의되면, train과 adding 단계가 존재 -> 보통 train과 adding 단계를 따로 하지 않는다.

---

* 2. Search based on FAISS index

![gbg](https://user-images.githubusercontent.com/59636424/137067018-1d857614-5e4b-4149-ae52-7416854397bd.PNG)

앞서, FAISS index가 만들어지면, inference 단계에서는 query가 들어오면 검색 후, 가장 가까운 cluster 방문하여 vector를 비교! -> top k를 뽑는다.

예로, 가장 가까운 10 cluster visit -> SQ8 base로 Search -> top k를 Search result로 도출

## 4. Scaling up with FAISS

### 4-1. FAISS Basics

1. 가장 단순한 인덱스 만들기(faiss.IndexFlatL2를 활용해 차원 정의하고 add한다)

-> pruning과 Scale quantiziation을 활용 안 하면 학습 X (index.train 존재 X)

2. 검색하기 (top k 정하기)

-> index.search를 이용

### 4-2. IVF with FAISS

* pruning -> IVF라는 이름으로 찾을 수 있다.

* **quantizer 만들기 -> cluster에서 거리를 잴 때 어떻게 잴 것인가?(clsuter와 query거리) => IndexFlatL2로 정의**

=> **faiss.IndexIVFFlat로 cluster 만들기**

=> IndexIVFFlat이 학습 데이터를 이용해 nlist만큼 cluster 만들기(train)

=> 학습 데이터 더해주기(add)

**더해주는 것은 다 더해줘야 한다! => 학습데이터 일부를 train은 할 수 있다!**

### 4-3. IVF-PQ with FAISS

SQ는 4 byte -> 1byte로, PQ는 768 dimension -> 100 byte로 줄일 수 있다. (많이 줄일 수 있다!)

### 4-4. Using GPU with FAISS

GPU 메모리에 따라, 벡터 갯수 제한!

# 2. Scaling up with FAISS (In-batch) 실습 정리!

**전에, 20개의 벡터 정도만 사용했다면 그 갯수를 FAISS로 늘려보자!!**

```python
import faiss

num_clusters = 16 # pruning을 위한 cluster 갯수
niter = 5 # 몇개 cluster 볼지
k = 5 # 최종 몇 개 가져올 지
```

* cluster를 위한 index만들기

```python
emb_dim = p_embs.shape[-1]
index_flat = faiss.IndexFlatL2(emb_dim)
```

* clustering 진행

```python
clus = faiss.Clustering(emb_dim, num_clusters)
clus.verbose = True
clus.niter = niter
clus.train(p_embs, index_flat)
```

* centroid 확보

```python
centroids = faiss.vector_float_to_array(clus.centroids)
centroids = centroids.reshape(num_clusters, emb_dim)
```

* quantizer 정의

```python
quantizer = faiss.IndexFlatL2(emb_dim)
quantizer.add(centroids) 
```

quantizer로 clsuter와 query거리 재는 법을 L2를 이용해 정의

* SQ8 + IVF indexer 진행 (Inverted File 만들기 -> quantizer로 cluster 만들기)

```python
indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
indexer.train(p_embs)
indexer.add(p_embs)
```

-> quantizer.ntotal = num_clusters

-> quantizer.d = emb_dim

* Search using indexer

**search를 통해 q_embs로 top k를 구한다.**

```python
D, I = indexer.search(q_embs, k)
```

-> D: 쿼리와의 거리

-> I: 검색된 벡터의 인덱스
