import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

name = 'unfair_identification_(klue_bert_base+MLP)_ver1_1차'
# ✅ KLUE/BERT 토크나이저 및 모델 로드
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)

# ✅ 저장할 디렉토리 설정 (폴더 없으면 생성)
save_path = f"E:/Model/ver2/{name}/"
os.makedirs(save_path, exist_ok=True)
model_file = os.path.join(save_path, "klue_bert_mlp.pth")

class BertMLPClassifier(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", hidden_size=256):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)  # 불공정(1) 확률을 출력
        self.sigmoid = nn.Sigmoid()  # 확률값으로 변환

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 사용
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)  # 0~1 확률값 반환


directory_path = './Data_Analysis/Data_ver2/unfair_data/'
files_to_merge = [f for f in os.listdir(directory_path) if 'preprocessing' in f and f.endswith('.csv')]
merged_df = pd.DataFrame()
for file in files_to_merge:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)
    print("*"*50)
    print(f'{file}')
    print(f'len-{len(df)}')
    print(f'--NaN-- \n {df.isna().sum()}')
    merged_df = pd.concat([merged_df, df], ignore_index=True)
print(f'merged_df: {len(merged_df)}')
x_temp, x_test, y_temp, y_test = train_test_split(
    merged_df["sentence"].tolist(),
    merged_df[["unfair_label", "article_number"]],  # DataFrame으로 두 열 선택
    test_size=0.1,
    random_state=42,
    stratify=merged_df[["article_number", "unfair_label"]],  # 두 개의 컬럼을 기준으로 stratify
    shuffle=True
)

# y_temp에서 'unfair_label'과 'article_number'를 분리
y_temp_labels = y_temp["unfair_label"]
y_temp_articles = y_temp["article_number"]

# 두 번째 Train/Val 데이터 분할 (8:2 비율)
# stratify에 article_number와 unfair_label 결합하여 두 기준을 동시에 고려하도록 함
X_train, X_val, y_train, y_val = train_test_split(
    x_temp,
    y_temp_labels,  # unfair_label만 사용
    test_size=0.2,
    random_state=42,
    stratify=y_temp[["article_number", "unfair_label"]],  # 두 기준을 동시에 stratify
    shuffle=True
)

print(f'Length of X_train (train data): {len(X_train)}')
print(f'Length of y_train (train labels): {len(y_train)}')
print(f'Length of X_val (validation data): {len(X_val)}')
print(f'Length of y_val (validation labels): {len(y_val)}')
print(f'Length of x_test (test data): {len(x_test)}')
print(f'Length of y_test (test labels): {len(y_test)}')

# ✅ 토큰화 및 텐서 변환 함수
def tokenize_data(sentences, tokenizer, max_length=256):
    encoding = tokenizer(
        sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]

# ✅ 훈련 및 검증 데이터 토큰화
X_train_ids, X_train_mask = tokenize_data(X_train, tokenizer)
X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # [batch, 1] 형태
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)



# ✅ 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMLPClassifier().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss 사용
optimizer = optim.Adam(model.parameters(), lr=0.00002)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# ✅ 모델 학습 함수 (Early Stopping 적용, 저장 X)
from torch.utils.data import DataLoader, TensorDataset

# ✅ 배치 크기 설정
batch_size = 16

# ✅ 데이터셋 & 데이터로더 설정
train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#######################################################################################################################
#######################################################################################################################


import matplotlib.pyplot as plt
# ✅ 모델 학습 함수 수정
# ✅ 손실 그래프 저장 함수
def plot_loss_curve(train_losses, val_losses, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Train Loss 축
    ax1.plot(train_losses, label="Train Loss", marker="o", color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Train Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True)

    # Validation Loss 축
    ax2 = ax1.twinx()
    ax2.plot(val_losses, label="Validation Loss", marker="o", color="red")
    ax2.set_ylabel("Validation Loss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # 제목 및 범례 추가
    plt.title("Training and Validation Loss")
    fig.tight_layout()
    plt.savefig(save_path)  # 이미지로 저장
    print(f"✅ Loss 그래프 저장 완료: {save_path}")
    plt.close()

# ✅ 모델 학습 함수 수정 (손실 그래프 추가)
def train_model(model, train_loader, val_loader, epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []
    best_model_state = None  # Best model weights 저장

    # ✅ ReduceLROnPlateau 스케줄러 추가 (patience=2, factor=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # ✅ 배치 학습 적용
        for X_batch, mask_batch, y_batch in train_loader:
            X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, mask_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # 단순 배치 평균
        train_loss = total_loss / len(train_loader)  # 배치 평균으로 통일

        # ✅ 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in val_loader:
                X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
                val_outputs = model(X_batch, mask_batch)
                val_loss += criterion(val_outputs, y_batch).item()

        val_loss /= len(val_loader)

        # ✅ 손실 저장
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # ✅ ReduceLROnPlateau 적용
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ✅ Early Stopping 체크
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Best model weights loaded into the model")

    # ✅ 손실 그래프 저장
    loss_plot_path = os.path.join(save_path, "loss_curve.png")
    plot_loss_curve(train_loss_list, val_loss_list, loss_plot_path)


# ✅ 모델 학습 실행 (손실 그래프 추가됨)
train_model(model, train_loader, val_loader, epochs=1000, patience=10)
torch.save(model.state_dict(), model_file)
print(f"✅ 모델 저장 완료: {model_file}")
#######################################################################################################################
#######################################################################################################################


# ✅ 불공정 조항 예측 함수 (수정 없음)
def predict_unfair_clause(c_model, sentence, threshold=0.5):
    """계약서 문장이 불공정한지 여부를 확률로 예측 (threshold 사용)"""
    c_model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = c_model(inputs["input_ids"], inputs["attention_mask"])
        unfair_prob = output.item()
    return {
        "sentence": sentence,
        "unfair_probability": round(unfair_prob * 100, 2),  # 1(불공정) 확률
        "predicted_label": "위반" if unfair_prob >= threshold else "합법"
    }

# ✅ 모델 저장 (state_dict만 저장)
def load_trained_model(model_file):
    # ✅ 모델 객체를 새로 생성한 후 state_dict만 로드해야 함
    model = BertMLPClassifier().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"✅ 저장된 모델 로드 완료: {model_file}")
    return model

# ✅ 모델 로드 실행
loaded_model = load_trained_model(model_file)

"""
import os, sys
sys.path.append(os.path.abspath("./AI"))
import threshold_settings as ts
threshold= ts.find_threshold(loaded_model, train_loader=train_loader, val_loader=val_loader, use_train=False, device=device)
최적 임계값: 0.5003
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = []
y_true = []
threshold = 0.5011
print(f'test: {len(x_test)}')
for sentence, label in zip(x_test, y_test["unfair_label"]):
    result = predict_unfair_clause(loaded_model,sentence,threshold)
    if result['predicted_label'] != f"{'위반' if label == 1 else '합법'}":
        print(f"📝 계약 조항: {result['sentence']}")
        print(f"🔍 판별 결과: {result['predicted_label']} (위반 확률: {result['unfair_probability']}%)")
        print(f"✅ 정답: {'위반' if label == 1 else '합법'}")
        print("-" * 50)
    y_pred.append(1 if result['unfair_probability'] >= threshold else 0)
    y_true.append(label)

# ✅ 성능 지표 계산
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# ✅ 성능 지표 출력
print("📊 테스트 데이터 성능 평가 결과 📊")
print(f"📌 최적 임계값: {threshold:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")