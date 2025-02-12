from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import torch
import re

# 1. KoBART 모델과 토크나이저 로드
model_name = 'gogamza/kobart-summarization'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# 2. 데이터셋 로드 (CSV 파일 기준)
dataset = load_dataset('csv', data_files={'full_data': 'combined_summary_data_v2.csv'})

# 3. 데이터셋 분할 (80% 훈련, 20% 테스트)
dataset = dataset['full_data'].train_test_split(test_size=0.2)

# 4. 토크나이즈 함수 정의
def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['summary']
    
    # 입력과 출력 텍스트를 토크나이즈
    model_inputs = tokenizer(inputs, max_length=1024, padding=True, truncation=True, return_tensors="pt")  # padding, truncation, return_tensors 추가
    labels = tokenizer(targets, max_length=256, padding=True, truncation=True, return_tensors="pt")  # padding, truncation, return_tensors 추가
    
    # 모델이 필요로 하는 라벨 포맷에 맞게 조정
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 5. 데이터셋 전처리 적용
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 6. 트레이닝 파라미터 설정
training_args = Seq2SeqTrainingArguments(
    output_dir='./kobart_contract_summary',
    evaluation_strategy='epoch',  # 평가 주기를 'epoch'으로 설정
    save_strategy='epoch',  # 저장 주기를 'epoch'으로 설정하여 일치시킴
    learning_rate=5e-5,  # 적절한 learning rate 설정
    per_device_train_batch_size=4,  # GPU 메모리에 맞게 배치 크기 조정
    per_device_eval_batch_size=4,  # 평가 배치 크기
    weight_decay=0.01,  # L2 규제
    save_total_limit=3,  # 최대 3개의 체크포인트만 저장
    num_train_epochs=5,  # 적절한 epoch 수 설정 (과적합 방지)
    predict_with_generate=True,  # 예측 시 생성 방식 사용
    load_best_model_at_end=True,  # 훈련 후 최고 성능 모델을 로드
    fp16=False,  # GPU 사용 시 mixed precision 학습
)

# 7. 트레이너 초기화
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# 8. 파인튜닝 시작
trainer.train()

# 9. 파인튜닝된 모델 저장
model.save_pretrained('C:/Model/kobart_finetuning')
tokenizer.save_pretrained('C:/Model/kobart_finetuning')

# 10. 예측을 수행하면서 generate 함수의 파라미터 추가
def generate_summary(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    
    output = model.generate(
        inputs['input_ids'],  # 입력 텍스트의 token ids
        max_length=300,  # 생성할 최대 길이
        num_beams=5,  # 빔 서치
        length_penalty=1.5,  # 길이 페널티로 긴 텍스트 유도
        early_stopping=True,  # 예측을 너무 일찍 끝내지 않도록 설정
        no_repeat_ngram_size=2  # 중복되는 n-gram 방지
    )
    
    # 예측된 텍스트 디코딩
    decoded_summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_summary

# 11. 불필요한 문자를 제거하는 함수 (일본어, 특수문자 등 필터링)
def clean_text(text):
    # 일본어 문자와 특수문자 제거 (한글, 영어, 숫자만 남기기)
    cleaned_text = re.sub(r'[^\w\s가-힣]', '', text)  # 한글, 영어, 숫자 외의 문자 제거
    return cleaned_text

# 12. 테스트 데이터에 대해 예측 수행
for i in range(5):  # 예시로 상위 5개의 예측 결과만 출력
    original_text = tokenized_datasets['test'][i]['input']
    predicted_summary = generate_summary(model, tokenizer, original_text)  # 위 함수로 예측 수행
    
    # 예측된 요약에서 불필요한 문자를 제거
    cleaned_summary = clean_text(predicted_summary)
    
    print(f"원본 텍스트: {original_text}")
    print(f"예측된 요약: {cleaned_summary}")
    print("="*50)
