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

## 2. prepare_validation_features 함수


```python
def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])): 
        sequence_ids = tokenized_examples.sequence_ids(i) #대부분 sequence_ids length는 384(max_seq_length)
        context_index = 1

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index]) # examples['id'] 삽입

        # context 부분의 offset mapping만 넣기
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples #dict_keys(['attention_mask', 'example_id', 'input_ids', 'offset_mapping', 'token_type_ids'])
```

## 3. post_processing_function 함수

```python
def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples, # The non-preprocessed dataset 
        features=features, # The processed dataset
        predictions=predictions, # The predictions of the model: two arrays containing the start logits and the end logits respectively
        version_2_with_negative=False, #기본 데이터 세트에 답변이 없는 예제가 포함되어 있는지 여부
        n_best_size=n_best_size, # 답변을 찾을 때 생성할 n-best 예측의 총 수
        max_answer_length=max_answer_length,
        null_score_diff_threshold=0.0,
        output_dir=training_args.output_dir,
        is_world_process_zero=trainer.is_world_process_zero(), # huggingface의 postprocess_qa_predictions에는 아예 없고 baseline에도 없음
    )
    
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
```
