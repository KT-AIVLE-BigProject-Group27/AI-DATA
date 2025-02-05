import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from datasets import Dataset
# 경로 설정: summarization_models 모듈(모델 생성 함수 포함)
sys.path.append(os.path.abspath("./AI"))
from summarization_models import get_kobart_model

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-summarization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################
# (0) 데이터 전처리 및 토큰화 함수
##############################################
def tokenize_function(example, tokenizer, max_input_length=512, max_target_length=128):
    # 입력 텍스트 토큰화
    model_inputs = tokenizer(example["text"], max_length=max_input_length, truncation=True)
    # 요약문(라벨) 토큰화 (target tokenizer 사용)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


##############################################
# (1) 평가 지표 계산 함수 (ROUGE 기반)
##############################################
def compute_metrics_fn(eval_pred, tokenizer):
    """예측 결과와 정답을 디코딩한 후 ROUGE 점수를 계산"""
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    return result

##############################################
# (2) 데이터 전처리 함수 정의 (모델 입력에 맞게 텍스트와 요약 토큰화)
##############################################
def preprocess_function(examples):
    # 입력 텍스트를 토큰화 (최대 길이 2048)
    model_inputs = tokenizer(examples["text"],max_length=1024,truncation=True,padding="max_length")
    # 요약 텍스트를 토큰화 (최대 길이 512)
    labels = tokenizer(examples["summary"],max_length=256,truncation=True,padding="max_length")
    # 모델의 타깃 레이블로 토큰화된 요약 추가
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 모델 결과 저장 및 비교를 위한 기본 경로
base_save_path = os.path.join("E:/Model/ver2", "summarization_comparison")
os.makedirs(base_save_path, exist_ok=True)

# 데이터 로드 (CSV 파일, 컬럼: "input", "summary")

directory_path = r'./Data_Analysis/Data_ver2/summary_data/'
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
dfs = []
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
merged_df['article'] = merged_df['input'].apply(lambda x: x.split('[')[0].strip())

import re

# 공백 2개 이상을 1개로 변환하는 함수
def clean_whitespace(text):
    text = text.replace('“','').replace('”','').replace("'",'').replace('"','')
    return re.sub(r'\s{2,}', ' ', text.strip())

# 'input'과 'summary' 컬럼에 적용
merged_df['input'] = merged_df['input'].apply(clean_whitespace)
merged_df['summary'] = merged_df['summary'].apply(clean_whitespace)


# ✅ 'input'과 'summary'의 토큰 길이 계산
merged_df["input_token_length"] = merged_df["input"].apply(lambda x: len(tokenizer.encode(x)))
merged_df["summary_token_length"] = merged_df["summary"].apply(lambda x: len(tokenizer.encode(x)))

# ✅ 토큰 기준 최대 길이 출력
max_input_token_length = merged_df["input_token_length"].max()
max_summary_token_length = merged_df["summary_token_length"].max()

print(f"📌 최대 input 토큰 길이: {max_input_token_length}")
print(f"📌 최대 summary 토큰 길이: {max_summary_token_length}")

# ✅ 1024 토큰 이상인 데이터 확인
long_token_texts = merged_df[merged_df["input_token_length"] > 1024]
print(long_token_texts)


# dataset은 "train" 키로 로드됨; Train/Test Split (예: 90% train, 10% test)
X_train,X_val,y_train,y_val = train_test_split(merged_df['input'],merged_df['summary'],test_size=0.2, random_state=42,stratify=merged_df['article'] )
train_data = pd.DataFrame({'text': X_train, 'summary': y_train})
val_data = pd.DataFrame({'text': X_val, 'summary': y_val})
print(f'X_train: {len(X_train)}, X_val: {len(X_val)}')

# 모델 리스트: (모델명, 생성 함수)
models_info = [
    ("kobart", get_kobart_model),
]

# 학습 파라미터 설정
num_train_epochs = 1000
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
learning_rate = 2e-5

# 각 모델에 대해 순차적으로 학습 및 평가
for model_label, get_model_fn in models_info:
    print("\n" + "=" * 50)
    print(f"모델: {model_label} 학습 시작")

    # 1. 모델 저장 경로 설정 및 디렉토리 생성
    cur_save_path = os.path.join(base_save_path, model_label)
    os.makedirs(cur_save_path, exist_ok=True)

    # 2. 모델과 토크나이저 로드
    model, tokenizer = get_model_fn()
    model.to(device)

    # 3. 데이터셋 변환 (pandas → Hugging Face Dataset)
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # 5. 데이터 토큰화 (batched=True 적용, 기존 컬럼 제거)
    tokenized_train = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess_function, batched=True, remove_columns=val_dataset.column_names
    )

    def compute_metrics(eval_pred):
        return compute_metrics_fn(eval_pred, tokenizer)

    # 6. TrainingArguments 설정 (학습 관련 인자들 지정)
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cur_save_path, "results"),
        evaluation_strategy="epoch",
        save_strategy="epoch",  # 매 epoch마다 저장
        save_total_limit=3,
        load_best_model_at_end=True,  # 🔹 가장 좋은 모델 저장 (restore_best_weights 포함)
        metric_for_best_model="eval_loss",  # 🔹 평가 기준 (eval_loss가 가장 낮을 때 저장)
        greater_is_better=False,  # 🔹 Loss는 낮을수록 좋음
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=0.01,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True if device.type == "cuda" else False,
        logging_dir=os.path.join(cur_save_path, "logs"),
        remove_unused_columns=False,  # "No columns" 에러 방지를 위해 사용
    )

    # 8. Seq2SeqTrainer 설정 (모델, 데이터셋, 토크나이저, 평가 함수 등 등록)
    from transformers import EarlyStoppingCallback

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]  # 🔹 Early Stopping 적용
    )

    # 9. 모델 학습 실행
    trainer.train()

    # 🔹 추가 1: Loss 데이터 저장 (CSV 저장)
    loss_history = pd.DataFrame(trainer.state.log_history)
    loss_history.to_csv(os.path.join(cur_save_path, "loss_history.csv"), index=False)

    # 🔹 추가 2: Loss 그래프 저장
    plt.figure(figsize=(6, 4))
    train_losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]

    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(eval_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_label} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cur_save_path, "loss_curve.png"))
    plt.close()

    # 10. 학습 후 평가 실행 및 결과 출력
    eval_results = trainer.evaluate()
    print(f"[{model_label}] 평가 결과:")
    print(eval_results)

    # 11. 학습된 모델과 토크나이저 저장
    model_save_path = os.path.join(cur_save_path, f"{model_label}_summarization")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"✅ 모델 및 토크나이저 저장 완료: {model_save_path}")


# 모델 리스트 (모델명과 생성 함수)
models_info = [
    ("kobart", get_kobart_model),
]
results = []  # 각 모델 평가 지표 저장
base_save_path = "E:/Model/ver2/summarization_comparison/"  # 저장된 모델들이 위치한 기본 경로 (예시)

for model_label, get_model_fn in models_info:
    print("\n" + "=" * 50)
    print(f"모델: {model_label} 테스트 시작")

    # 각 모델 전용 save 경로
    cur_save_path = os.path.join(base_save_path, model_label)
    model_save_path = os.path.join(cur_save_path, f"{model_label}_summarization")

    # 모델과 토크나이저 생성 (모델은 raw logits/생성 결과를 반환하도록 구현되어 있어야 함)
    model, tokenizer = get_model_fn()

    # 저장된 모델과 토크나이저 로드 (클래스 메서드 방식으로 호출)
    model = model.__class__.from_pretrained(model_save_path)
    tokenizer = type(tokenizer).from_pretrained(model_save_path)
    model.to(device)

    # 3. 데이터셋 변환 (pandas → Hugging Face Dataset)
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # 5. 데이터 토큰화 (batched=True 적용, 기존 컬럼 제거)
    tokenized_train = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess_function, batched=True, remove_columns=val_dataset.column_names
    )
    # Trainer 평가를 위한 TrainingArguments (평가만 진행하므로 간단한 설정)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./dummy_output",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        evaluation_strategy="no",  # 평가만 수행
        remove_unused_columns = False
    )

    # Trainer 초기화 (compute_metrics은 생략하거나 추가 가능)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )

    # 평가 실행
    eval_results = trainer.evaluate()
    print(f"[{model_label}] 평가 결과:")
    print(eval_results)

    # 결과 저장: 여기서는 eval_loss와 (있다면) ROUGE-L 점수를 저장
    result_entry = {
        "Model": model_label,
        "Eval_Loss": eval_results.get("eval_loss", np.nan),
        "RougeL": eval_results.get("eval_rougeL", np.nan)
    }
    results.append(result_entry)


    # 모델 평가 및 예측 수행
    predictions = trainer.predict(tokenized_val)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode([label for label in predictions.label_ids], skip_special_tokens=True)
    original_inputs = val_data["text"].tolist()
    results_df = pd.DataFrame({
        "Input": original_inputs,
        "Predicted_Summary": decoded_preds,
        "Ground_Truth_Summary": decoded_labels
    })
    results_csv_path = f"{cur_save_path}/{model_label}_summarization_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ 모델 입력/출력 결과 저장 완료: {results_csv_path}")

# 성능 비교 시각화 (모델별 평가 지표)
results_df = pd.DataFrame(results)
print("\n전체 모델 성능 비교:")
print(results_df)

# 각 평가 지표에 대해 bar chart 생성
metrics = ["Eval_Loss", "RougeL"]
num_metrics = len(metrics)
fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
if num_metrics == 1:
    axs = [axs]
for i, metric in enumerate(metrics):
    axs[i].bar(results_df["Model"], results_df[metric], color="skyblue")
    axs[i].set_title(metric)
    axs[i].set_ylim([0, results_df[metric].max() * 1.2])
    for j, value in enumerate(results_df[metric]):
        axs[i].text(j, value + 0.02 * results_df[metric].max(), f"{value:.2f}", ha="center", va="bottom", fontsize=10)
plt.suptitle("모델별 성능 비교", fontsize=16)
comparison_plot_file = os.path.join(base_save_path, "model_comparison.png")
plt.tight_layout()
plt.savefig(comparison_plot_file)
plt.show()
