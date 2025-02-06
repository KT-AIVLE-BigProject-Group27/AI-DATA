import os
import sys
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    AutoTokenizer
)
import evaluate
from datasets import Dataset
from functools import partial

# 경로 설정: summarization_models 모듈(모델 생성 함수 포함)
sys.path.append(os.path.abspath("./AI"))
from summarization_models import get_kobart_model

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################
# (0) 토큰화 및 전처리 함수 (입력 및 요약)
##############################################
def preprocess_function(examples, tokenizer, max_input_length=1024, max_target_length=256):
    """
    examples: dict, keys "text" and "summary"
    tokenizer: Hugging Face 토크나이저
    """
    # 입력 텍스트 토큰화
    model_inputs = tokenizer(examples["text"],
                             max_length=max_input_length,
                             truncation=True,
                             padding="max_length")
    # 요약 텍스트 토큰화
    labels = tokenizer(examples["summary"],
                       max_length=max_target_length,
                       truncation=True,
                       padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


##############################################
# (1) 평가 지표 계산 함수 (ROUGE 기반)
##############################################
def compute_metrics_fn(eval_pred, tokenizer):
    """
    eval_pred: (predictions, labels)
    tokenizer: Hugging Face 토크나이저
    """
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred

    # 예측값 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # -100을 tokenizer의 pad_token_id로 대체한 후 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds,
                           references=decoded_labels,
                           use_stemmer=True)
    # 결과값 스케일 조정
    result = {key: value * 100 for key, value in result.items()}
    return result


##############################################
# (2) 데이터 로드 및 전처리
##############################################
# 기본 모델 저장 경로
base_save_path = os.path.join("E:/Model/ver2", "summarization_comparison")
os.makedirs(base_save_path, exist_ok=True)

# CSV 데이터 로드 (컬럼: "input", "summary")
directory_path = r'./Data_Analysis/Data_ver2/summary_data/'
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(directory_path, file)) for file in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)

# article 컬럼 생성 (예시: 'input'의 앞부분 기준)
merged_df['article'] = merged_df['input'].apply(lambda x: x.split('[')[0].strip())


# 텍스트 클린징 함수: 불필요한 따옴표 제거 및 공백 정리
def clean_whitespace(text):
    text = text.replace('“', '').replace('”', '').replace("'", '').replace('"', '')
    return re.sub(r'\s{2,}', ' ', text.strip())


merged_df['input'] = merged_df['input'].apply(clean_whitespace)
merged_df['summary'] = merged_df['summary'].apply(clean_whitespace)

# 토큰 길이 측정을 위해 임시로 토크나이저 생성 (모델 로드 전)
# ※ get_kobart_model()로 모델과 토크나이저를 로드하면 되지만, 토큰 길이 확인을 위해 임시 토크나이저 사용
temp_tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-summarization")
merged_df["input_token_length"] = merged_df["input"].apply(lambda x: len(temp_tokenizer.encode(x)))
merged_df["summary_token_length"] = merged_df["summary"].apply(lambda x: len(temp_tokenizer.encode(x)))

max_input_token_length = merged_df["input_token_length"].max()
max_summary_token_length = merged_df["summary_token_length"].max()
print(f"📌 최대 input 토큰 길이: {max_input_token_length}")
print(f"📌 최대 summary 토큰 길이: {max_summary_token_length}")

# (옵션) 1024 토큰 이상인 데이터 확인
long_token_texts = merged_df[merged_df["input_token_length"] > 1024]
print(long_token_texts)


# 추가 텍스트 정제: 유니코드 정규화 및 특수 문자/특수 숫자 변환
def clean_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', '', text)
    return text


def convert_special_numbers(text):
    special_numbers = {
        '①': '(1)', '②': '(2)', '③': '(3)', '④': '(4)', '⑤': '(5)',
        '⑥': '(6)', '⑦': '(7)', '⑧': '(8)', '⑨': '(9)', '⑩': '(10)'
    }
    for k, v in special_numbers.items():
        text = text.replace(k, v)
    return text


merged_df['input'] = merged_df['input'].apply(lambda x: convert_special_numbers(clean_text(x)))
merged_df['summary'] = merged_df['summary'].apply(clean_text)

# Train/Test Split (예: 80% train, 20% validation), stratify 기준은 'article'
X_train, X_val, y_train, y_val = train_test_split(
    merged_df['input'],
    merged_df['summary'],
    test_size=0.2,
    random_state=42,
    stratify=merged_df['article']
)
train_data = pd.DataFrame({'text': X_train, 'summary': y_train})
val_data = pd.DataFrame({'text': X_val, 'summary': y_val})
print(f'X_train: {len(X_train)}, X_val: {len(X_val)}')

##############################################
# (3) 학습 및 평가 루프
##############################################
# 모델 리스트: (모델명, 생성 함수)
models_info = [
    ("kobart", get_kobart_model),
]

# 학습 하이퍼파라미터 설정
num_train_epochs = 100
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
learning_rate = 3e-5

# 각 모델에 대해 학습 및 평가 수행
for model_label, get_model_fn in models_info:
    print("\n" + "=" * 50)
    print(f"모델: {model_label} 학습 시작")

    # 1. 모델 저장 경로 생성
    cur_save_path = os.path.join(base_save_path, model_label)
    os.makedirs(cur_save_path, exist_ok=True)

    # 2. 모델과 토크나이저 로드 및 device 할당
    model, tokenizer = get_model_fn()
    model.to(device)

    # 3. Hugging Face Dataset으로 변환
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # 4. 데이터 토큰화 (함수에 tokenizer를 인자로 전달)
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    ).with_format("torch")
    tokenized_val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    ).with_format("torch")

    # 5. compute_metrics 함수에 tokenizer 전달 (partial 사용)
    compute_metrics = partial(compute_metrics_fn, tokenizer=tokenizer)

    # 6. TrainingArguments 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cur_save_path, "results"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",  # ROUGE 기준 최적 모델 선택
        greater_is_better=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=0.01,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True if device.type == "cuda" else False,
        logging_dir=os.path.join(cur_save_path, "logs"),
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
    )

    # 7. Seq2SeqTrainer 초기화
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8. (옵션) 데이터셋의 포맷을 torch로 명시 (이미 with_format("torch")를 사용했으므로 생략 가능)
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # 9. 모델 학습 실행
    trainer.train()

    # 10. 학습 로그 저장 (CSV 및 Loss Curve 그림)
    loss_history = pd.DataFrame(trainer.state.log_history)
    loss_history.to_csv(os.path.join(cur_save_path, "loss_history.csv"), index=False)

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

    # 11. 모델 평가 실행
    eval_results = trainer.evaluate()
    print(f"[{model_label}] 평가 결과:")
    print(eval_results)

    # 12. 학습된 모델과 토크나이저 저장
    model_save_path = os.path.join(cur_save_path, f"{model_label}_summarization")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"✅ 모델 및 토크나이저 저장 완료: {model_save_path}")

##############################################
# (4) 저장된 모델로 테스트 및 예측 수행, 평가 결과 저장
##############################################
results_summary = []  # 모델별 평가 지표 저장
for model_label, get_model_fn in models_info:
    print("\n" + "=" * 50)
    print(f"모델: {model_label} 테스트 시작")

    # 저장된 모델 경로
    cur_save_path = os.path.join(base_save_path, model_label)
    model_save_path = os.path.join(cur_save_path, f"{model_label}_summarization")

    # 저장된 모델과 토크나이저 로드
    model, tokenizer = get_model_fn()
    model = model.__class__.from_pretrained(model_save_path)
    tokenizer = type(tokenizer).from_pretrained(model_save_path)
    model.to(device)

    # Dataset으로 변환 후 토큰화
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    tokenized_val = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Trainer 평가를 위한 간단한 TrainingArguments 설정
    eval_training_args = Seq2SeqTrainingArguments(
        output_dir="./dummy_output",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        evaluation_strategy="no",
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_training_args,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )

    # 평가 실행
    eval_results = trainer.evaluate()
    print(f"[{model_label}] 평가 결과:")
    print(eval_results)

    # 결과 저장: eval_loss 및 (있다면) ROUGE-L 점수
    result_entry = {
        "Model": model_label,
        "Eval_Loss": eval_results.get("eval_loss", np.nan),
        "RougeL": eval_results.get("eval_rougeL", np.nan)
    }
    results_summary.append(result_entry)

    # 모델 예측 수행 및 결과 CSV 저장
    predictions = trainer.predict(tokenized_val)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    original_inputs = val_data["text"].tolist()
    results_df = pd.DataFrame({
        "Input": original_inputs,
        "Predicted_Summary": decoded_preds,
        "Ground_Truth_Summary": decoded_labels
    })
    results_csv_path = os.path.join(cur_save_path, f"{model_label}_summarization_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 모델 입력/출력 결과 저장 완료: {results_csv_path}")

# 모델별 평가 지표 요약 및 시각화
results_df = pd.DataFrame(results_summary)
print("\n전체 모델 성능 비교:")
print(results_df)

# 평가 지표별 bar chart 생성
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
        axs[i].text(j, value + 0.02 * results_df[metric].max(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=10)
plt.suptitle("모델별 성능 비교", fontsize=16)
plt.tight_layout()
comparison_plot_file = os.path.join(base_save_path, "model_comparison.png")
plt.savefig(comparison_plot_file)
plt.show()
