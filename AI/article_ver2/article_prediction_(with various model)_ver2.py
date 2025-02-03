import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 경로 설정: train_def 모듈(모델 생성 및 train_model 함수 포함)
sys.path.append(os.path.abspath("./AI"))
from prediction_models import (
    get_KLUE_BERT_model,
    get_KoBERT_model,
    get_KoELECTRA_model,
    get_KLUE_Roberta_model,
    get_KoSBERT_model,
    get_XLMRoberta_model,
    train_model
)


##############################################
# (0) 공통 함수 정의
##############################################
# 토큰화 함수 (모델마다 tokenizer가 다르므로, 각 모델의 tokenizer를 사용)
def tokenize_data(sentences, tokenizer, max_length=256):
    encoding = tokenizer(
        sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]


# 모델 예측 함수 (각 모델마다 tokenizer가 다르므로, 인자로 받음)
def predict_article(model, tokenizer, sentence):
    model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    """
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    """
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = torch.argmax(output).item()
        predicted_article = idx_to_article[predicted_idx]
    return {
        "sentence": sentence,
        "predicted_article": predicted_article
    }


# Warm-up scheduler 함수
def warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


##############################################
# (1) 데이터 로드 및 전처리 (공통)
##############################################
# 모델 결과 저장 및 비교를 위한 기본 경로
name = 'article_prediction_ver2_비교'
base_save_path = os.path.join("E:/Model/ver2", name)
os.makedirs(base_save_path, exist_ok=True)

# 계약서 조항 제목 매핑
article_to_title = {
    '1': '[목적]', '2': '[기본원칙]', '3': '[공정거래 준수 및 동반성장 지원]', '4': '[상품의 납품]', '5': '[검수기준 및 품질검사]',
    '6': '[납품대금 지급 및 감액금지]', '6-2': '[공급원가 변동에 따른 납품 가격의 조정]', '7': '[상품의 반품]', '8': '[판매장려금]',
    '9': '[판촉사원 파견 등]', '10': '[서비스 품질유지]', '11': '[판촉행사 참여 등]', '12': '[매장 위치 및 면적 등]',
    '12-2': '[매장이동 기준 등의 사전 통지]', '13': '[기타 비용의 사전 통지]', '14': '[경영정보 제공 요구 금지]',
    '15': '[보복조치의 금지]', '16': '[각종 불이익 제공 금지 등]', '17': '[손해배상]', '18': '[지식재산권 등]',
    '19': '[상표관련특약]', '20': '[제조물책임]', '21': '[권리ㆍ의무의 양도금지]', '22': '[통지의무]', '23': '[비밀유지]',
    '24': '[계약해지]', '25': '[상계]', '26': '[계약의 유효기간 및 갱신]', '26-2': '[계약의 갱신 기준 등의 사전 통지]',
    '27': '[분쟁해결 및 재판관할]', '28': '[계약의 효력]'
}

# 데이터 로드 (여러 CSV 파일 병합)
directory_path = './Data_Analysis/Data_ver2/unfair_data/'
files_to_merge = [f for f in os.listdir(directory_path) if 'preprocessing' in f and f.endswith('.csv')]
merged_df = pd.DataFrame()
for file in files_to_merge:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, df], ignore_index=True)
merged_df["article_number"] = merged_df["article_number"].astype(str)
print(f'merged_df: {len(merged_df)}')

merged_df["sentence"] = merged_df.apply(
    lambda row: f"{article_to_title.get(row['article_number'])} {row['sentence']}", axis=1
)

df_unfair = merged_df[merged_df["unfair_label"] == 1].reset_index(drop=True)

article_to_idx = {article: idx for idx, article in enumerate(df_unfair["article_number"].unique())}
idx_to_article = {idx: article for article, idx in article_to_idx.items()}

df_unfair["article_number"] = df_unfair["article_number"].map(article_to_idx)

# Train/Test 분할 (stratify: article_number, unfair_label)
x_temp, x_test, y_temp, y_test = train_test_split(
    df_unfair["sentence"].tolist(),
    df_unfair["article_number"],
    test_size=0.1,
    random_state=42,
    stratify=df_unfair["article_number"],
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    x_temp,
    y_temp,
    test_size=0.2,
    random_state=42,
    stratify=y_temp,
    shuffle=True
)

print(f'X_train: {len(X_train)}, X_val: {len(X_val)}, x_test: {len(x_test)}')

# DataLoader 구성
# (각 모델별로 토크나이저가 다르므로, DataLoader는 모델 학습 직전에 토큰화)
##############################################
# (2) 학습/테스트 루프: 5개 모델 순차 실행
##############################################
# 모델 리스트 (label과 생성 함수)
models_info = [
    ("KLUE-BERT", get_KLUE_BERT_model),
    ("KoBERT", get_KoBERT_model),
    ("KoELECTRA", get_KoELECTRA_model),
    ("KLUE-RoBERTa", get_KLUE_Roberta_model),
    ("KoSBERT", get_KoSBERT_model),
    ("XLM-RoBERTa", get_XLMRoberta_model)
]

# 학습 파라미터
batch_size = 16
learning_rate = 0.00002
num_epochs = 1000  # 데모용: 실제 실험에서는 epochs=1000, patience=10 등으로 설정할 수 있음
patience = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 성능 지표를 저장할 리스트
results = []

# 모델별 학습 및 평가 루프
for model_label, get_model_fn in models_info:
    print("\n" + "=" * 50)
    print(f"모델: {model_label} 학습 시작")

    # 각 모델 전용 save 경로 및 파일 설정
    cur_save_path = os.path.join(base_save_path, model_label)
    os.makedirs(cur_save_path, exist_ok=True)
    model_file = os.path.join(cur_save_path, f"{model_label}_mlp.pth")

    # 모델과 토크나이저 생성
    model, tokenizer = get_model_fn(num_classes = len(idx_to_article), hidden_size=256)
    model = model.to(device)

    # 데이터 토큰화: 각 모델별 tokenizer 사용 (Train/Val)
    X_train_ids, X_train_mask = tokenize_data(X_train, tokenizer)
    X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
    val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 옵티마이저, 손실 함수, 스케줄러 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * 10
    num_warmup_steps = int(0.1 * num_training_steps)
    warmup_sched = warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)
    reduce_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # 학습 실행 (train_model 함수는 내부에서 손실 그래프 등도 저장)
    train_model(model, optimizer, criterion, device, train_loader, val_loader,
                warmup_sched, reduce_sched, num_warmup_steps, cur_save_path, model_file,
                epochs=num_epochs, patience=patience)

models_info = [
    ("KLUE-BERT", get_KLUE_BERT_model),
    ("KoBERT", get_KoBERT_model),
    ("KoELECTRA", get_KoELECTRA_model),
    ("KLUE-RoBERTa", get_KLUE_Roberta_model),
    ("KoSBERT", get_KoSBERT_model),
    ("XLM-RoBERTa", get_XLMRoberta_model)
]
for model_label, get_model_fn in models_info:
    print("\n" + "=" * 50)
    print(f"모델: {model_label} 학습 시작")

    # 각 모델 전용 save 경로 및 파일 설정
    cur_save_path = os.path.join(base_save_path, model_label)
    os.makedirs(cur_save_path, exist_ok=True)
    model_file = os.path.join(cur_save_path, f"{model_label}_mlp.pth")

    # 모델과 토크나이저 생성
    model, tokenizer = get_model_fn(num_classes = len(idx_to_article), hidden_size=256)
    model = model.to(device)
    # 모델 로드 (학습 시 best state 저장된 파일에서 로드)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # 테스트: x_test에 대해 예측 수행 (모델별 tokenizer 사용)
    y_pred = []
    y_true = []

    for sentence, label in zip(x_test, y_test["article_number"]):
        result = predict_article(model, tokenizer, sentence)
        predicted_label_idx = result['predicted_article']
        true_article = idx_to_article[label]

        y_pred.append(predicted_label_idx)
        y_true.append(true_article)

    # 성능 지표 계산
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=1)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)
    try:
        roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
    except ValueError:
        roc_auc = float('nan')

    print(f"\n[{model_label}] 테스트 성능:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # 결과 저장
    results.append({
        "Model": model_label,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

##############################################
# (3) 모델 성능 비교 시각화
##############################################
results_df = pd.DataFrame(results)
print("\n전체 모델 성능 비교:")
print(results_df)

# 바 차트로 비교: 각 성능 지표별로 subplot 생성
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
num_metrics = len(metrics)

fig, axs = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 5))
for i, metric in enumerate(metrics):
    axs[i].bar(results_df["Model"], results_df[metric], color='skyblue')
    axs[i].set_title(metric)
    axs[i].set_ylim([0, 1])
    for j, value in enumerate(results_df[metric]):
        axs[i].text(j, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
plt.suptitle("모델별 성능 비교", fontsize=16)
comparison_plot_file = os.path.join(base_save_path, "model_comparison.png")
plt.tight_layout()
plt.savefig(comparison_plot_file)
plt.show()
