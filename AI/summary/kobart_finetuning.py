from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import torch

# 1. KoBART 모델과 토크나이저 로드
model_name = 'gogamza/kobart-summarization'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# 2. 데이터셋 로드 (CSV 파일 기준)
dataset = load_dataset('csv', data_files={'train': 'train_data.csv', 'validation': 'val_data.csv'})

# 3. 토크나이즈 함수 정의
def preprocess_function(examples):
    inputs = examples['source']
    targets = examples['target']
    
    # 입력과 출력 텍스트를 토크나이즈
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=150, truncation=True)
    
    # 모델이 필요로 하는 라벨 포맷에 맞게 조정
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 4. 데이터셋 전처리 적용
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. 트레이닝 파라미터 설정
training_args = Seq2SeqTrainingArguments(
    output_dir='./kobart_contract_summary',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,  # GPU 사용 시 mixed precision 학습
)

# 6. 트레이너 초기화
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
)

# 7. 파인튜닝 시작
trainer.train()

# 8. 파인튜닝된 모델 저장
model.save_pretrained('./fine_tuned_kobart_contract')
tokenizer.save_pretrained('./fine_tuned_kobart_contract')
