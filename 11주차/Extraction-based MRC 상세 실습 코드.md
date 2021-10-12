# 1. MRC Practice 2 - Extraction-based MRC 상세 실습 코드

## 1. prepare_train_features 함수

```python
def prepare_train_features(examples):
  tokenized_examples = tokenizer(
      examples["question"],
      examples["context"],
      truncation = "only_second", #max_seq_length(384)까지 truncate한다!!
      max_length = max_seq_length,
      stride = doc_stride,
      return_overflowing_tokens=True, # max_seq_length 초과 시, 넘기기
      return_offsets_mapping=True, # 각 토큰에 대해서 (start char point, end char point) 정보 반환
      padding="max_length"
  )
  #tokenized_examples.keys(): dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])

  overflow_to_sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping") #[0,0] , 데이터 2개일 때, [0,0,1,1] -> 길이 짧으면 [0,0,1] 될 수도 있다.

  offset_mapping = tokenized_examples.pop('offset_mapping') #원본 context 내 몇번째 글자부터 몇번째 글자까지 해당하는지 확인

  tokenized_examples["start_positions"]=[]
  tokenized_examples["end_positions"]=[]

  for i, offsets in enumerate(offset_mapping):
    input_ids = tokenized_examples["input_ids"][i] # i번째 token 줄 받기
    cls_index = input_ids.index(tokenizer.cls_token_id) # tokenizer.cls_token_id(101) => CLS token은 다 첫번째 index에 존재 = 0

    sequence_ids = tokenized_examples.sequence_ids(i) #i번째 줄 CLS, SEP token -> None, 질문 token -> 0, context token -> 1의 list로 반환

    example_index = overflow_to_sample_mapping[i] # 0
    answers = examples["answers"][example_index] # batch_size만큼 들어오니 그 answer 중 example_index의 answer 추출

    answer_start_offset = answers['answer_start'][0] #answer의 시작점 추출
    answer_end_offset = answer_start_offset + len(answers['text'][0]) # 'text': ['리스트']로 들어오니 answer['texts'][0]을 해야 answer text 추출 가능!

    token_start_index = 0 # context 부분 시작점 추출
    while sequence_ids[token_start_index]!=1:
      token_start_index +=1

    token_end_index = len(input_ids) - 1 #context 끝부분 추출(보통 끝에는 padding이 있으므로 padding 전 부분 추출)
    while sequence_ids[token_end_index] !=1:
      token_end_index -= 1
    
    # context 범위 안에 벗어났을 경우
    if not (offsets[token_start_index][0] <= answer_start_offset and answer_end_offset <= offsets[token_end_index][1]):
      tokenized_examples["start_positions"].append(cls_index) # 벗어났으니 답이 없다는 뜻으로 0으로 채움
      tokenized_examples["end_positions"].append(cls_index)

    # contexst 범위 안에 있는 경우
    else:
      # offsets 길이보다는 작아야 한다!
      # 원래 answer의 시작 index보다 작거나 같은 offsets의 token_start_index점을 찾음
      # ==> 이렇게 answer의 시작점으로 token_start_index를 만듦(offset상)
      while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start_offset:
        token_start_index+=1
      tokenized_examples["start_positions"].append(token_start_index-1)

      # 같은 이유로 끝점으로 token_end_index를 만듦(offset상)
      while offsets[token_end_index][1]>=answer_end_offset:
        token_end_index-=1
      tokenized_examples["end_positions"].append(token_end_index+1)

  print(tokenized_examples["start_positions"],tokenized_examples["end_positions"])
  return tokenized_examples
```
