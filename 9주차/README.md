# Options

## train.py

|argument        |description|default|
|:--------------:|:----------|:------|
|--save_dir|모델 저장 경로 설정| ./best_models|
|--PLM|사용할 모델 선택(checkpoint)|klue/bert-base|
|--MLM_checkpoint  | MLM 모델 불러오기 |./best_models/klue-roberta-large-rtt-pem-mlm|
|--entity_flag| typed entity marker punct 사용 | False |
| --use_mlm | MaskedLM pretrained model 사용 유무 | False |
| --epochs | train epoch 횟수 지정 | 3 |
| --lr | learning rate 지정 | 5e-5 |
| --train_batch_size | train batch size 설정 | 16 |
| --warmup_steps | warmup step 설정 | 500 |
| --weigth_decay | weight decay 설정 | 0.01 |
| --evaluatoin_stratgey   | evaluation_strategy 설정 | steps |
| --ignore_mismatched | pretrained model load 시, mismatched size 무시 유무 | False|
| --eval_flag | validation data 사용 유무 | False |
| --eval_ratio | evalation data size ratio 설정 | 0.2 |
| --seed | random seed 설정 | 2 |
| --dotenv_path | 사용자 env 파일 경로 설정   | /opt/ml/wandb.env |
| --wandb_unique_tag | wandb tag 설정 | bert-base-high-lr |
| --entity_flag | 사용자 env 파일 경로 설정   | /opt/ml/wandb.env |
| --preprocessing_cmb | 데이터 전처리 방식 선택(0: 특수 문자 제거, 1: 특수 문자 치환, 2: date 보정, 3: 한글 띄워주기)| set ex: 0 1 2 |
| --mecab_flag | mecab을 활용한 형태소 분리 | False |
| --add_unk_token | unk token vocab에 저장 | False |
| --k_fold | Stratified K Fold 사용 | 0 |
| --adea_flag | adea 사용 유무 | False |
| --augmentation_flag | rtt augmentation dataset 사용 유무 | False |
| --model_type | 대,소분류 진행할 class 입력 | default |
| --model_name | custom 모델 입력 | None |





## inference.py

|argument  |description|default|
|:------------:|:----------|:------|
|--model_dir| 선택할 모델 경로 |./best_models|
|--PLM | 모델 checkpoint |klue/bert-base|
|--entity_flag | typed entity marker punct 사용 유무 |False|
|--preprocessing_cmb|데이터 전처리 방식 선택(0: 특수 문자 제거, 1: 특수 문자 치환, 2: date 보정, 3: 한글 띄워주기)| set ex: 0 1 2 |
|--mecab_flag | Mecab을 활용해 형태소를 분리 유무 | False |
| --add_unk_token | unk token vocab에 저장한 tokenizer 사용 유무 | False |
| --k_fold | Stratified K Fold 사용 | 0 |
| --model_type | 대,소분류 진행 유무 | False |
| --model_name | custom 모델 입력 | None |

## model_ensemble.py

|argument  |description|default|
|:------------:|:----------|:------|
|--dir| 앙상블할 모델 경로 선택 |all|

## train_mlm.py

**train.py의 --use_pem, --model_type 제외 동일**

### train.py에서 추가된 부분

|argument  |description|default|
|:------------:|:----------|:------|
|--use_pem| 데이터 전처리 방식 선택 유무 |False|
