import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

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

name = 'article_prediction_(klue_bert_base+MLP)_ver1_4차'
# ✅ KLUE/BERT 토크나이저 및 모델 로드
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)

# ✅ 저장할 디렉토리 설정 (폴더 없으면 생성)
save_path = f"E:/Model/ver2/{name}/"
os.makedirs(save_path, exist_ok=True)
model_file = os.path.join(save_path, "klue_bert_mlp.pth")

###############################################################################################################################################
# 데이터 로드 및 전처리
###############################################################################################################################################
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
merged_df["article_number"] = merged_df["article_number"].astype(str)
print(f'merged_df: {len(merged_df)}')

merged_df["sentence"] = merged_df.apply(
    lambda row: f"{article_to_title.get(row['article_number'])} {row['sentence']}", axis=1
)

df_unfair = merged_df[merged_df["unfair_label"] == 1].reset_index(drop=True)

article_to_idx = {article: idx for idx, article in enumerate(df_unfair["article_number"].unique())}
idx_to_article = {idx: article for article, idx in article_to_idx.items()}
df_unfair["article_number"] = df_unfair["article_number"].map(article_to_idx)
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
print(f'Length of X_train (train data): {len(X_train)}')
print(f'Length of y_train (train labels): {len(y_train)}')
print(f'Length of X_val (validation data): {len(X_val)}')
print(f'Length of y_val (validation labels): {len(y_val)}')
print(f'Length of x_test (test data): {len(x_test)}')
print(f'Length of y_test (test labels): {len(y_test)}')
def tokenize_data(sentences, tokenizer, max_length=256):
    encoding = tokenizer(
        sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]
X_train_ids, X_train_mask = tokenize_data(X_train, tokenizer)
X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
batch_size = 16
train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
num_articles = len(article_to_idx)
###############################################################################################################################################
# 모델 설계
###############################################################################################################################################
class BertMLPClassifier(nn.Module):
    def __init__(self, bert_model_name="klue/bert-base", hidden_size=256, num_classes=num_articles):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMLPClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002)





# Warm-up Scheduler 정의
def warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warm-up 단계: 학습률을 선형 증가
            return float(current_step) / float(max(1, num_warmup_steps))
        # Warm-up 종료 후 학습률 유지
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

# 파라미터 설정
num_epochs = 10
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 전체 스텝의 10%를 Warm-up으로 설정

# Warm-up 및 ReduceLROnPlateau 스케줄러 초기화
warmup_sched = warmup_scheduler(optimizer, num_warmup_steps, num_training_steps)
reduce_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
# Cosine Annealing
# 손실 그래프 저장 함수
def plot_loss_curve(train_losses, val_losses, lr_list, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train_losses, label="Train Loss", marker="o", color="blue")
    ax1.plot(val_losses, label="Validation Loss", marker="o", color="red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(lr_list, label="Learning Rate", marker="x", linestyle="dashed", color="green")
    ax2.set_ylabel("Learning Rate", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Training Loss, Validation Loss & Learning Rate")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 학습 함수
def train_model(model, train_loader, val_loader, epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []
    best_model_state = None
    current_step = 0
    lr_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training Loop
        for X_batch, mask_batch, y_batch in train_loader:
            X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, mask_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Warm-up 스케줄러 적용
            if current_step < num_warmup_steps:
                warmup_sched.step()
                for param_group in optimizer.param_groups:
                    print(f"Current Learning Rate: {param_group['lr']}")
            total_loss += loss.item()
            current_step += 1

        train_loss = total_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in val_loader:
                X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
                val_outputs = model(X_batch, mask_batch)
                val_loss += criterion(val_outputs, y_batch).item()
        val_loss /= len(val_loader)

        # ✅ 현재 학습률 저장
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)


        # Validation Loss 기록
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # ReduceLROnPlateau 스케줄러 적용 (Warm-up 이후)
        if current_step >= num_warmup_steps:
            reduce_sched.step(val_loss)
            print(f"ReduceLROnPlateau Adjusted Learning Rate: {optimizer.param_groups[0]['lr']}")

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print("✅ Best model weights loaded into the model")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)  # ✅ 최적 모델 복원
        print(f"🔄 Restored best model weights with val_loss {best_loss:.8f}")
    else:
        print("⚠️ Warning: No best model found, training ended without improvement.")

    loss_data = pd.DataFrame({
        "Epoch": list(range(1, len(train_loss_list) + 1)),
        "Train Loss": train_loss_list,
        "Validation Loss": val_loss_list,
        "Learning Rate": lr_list
    })
    # Loss 그래프 저장
    loss_csv_path = os.path.join(save_path, "loss_and_lr.csv")
    loss_data.to_csv(loss_csv_path, index=False)
    loss_plot_path = os.path.join(save_path, "loss_curve.png")
    plot_loss_curve(train_loss_list, val_loss_list, lr_list, loss_plot_path)
# 모델 학습 실행
train_model(model, train_loader, val_loader, epochs=200, patience=10)
torch.save(model.state_dict(), model_file)

###############################################################################################################################################
# 모델 Test
###############################################################################################################################################
def predict_article(c_model, sentence):
    c_model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = c_model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = torch.argmax(output).item()
        predicted_article = idx_to_article[predicted_idx]
    return {
        "sentence": sentence,
        "predicted_article": predicted_article
    }
def load_article_model(model_file):
    model = BertMLPClassifier().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"✅ 저장된 모델 로드 완료: {model_file}")
    return model
loaded_model = load_article_model(model_file)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print(f'test: {len(x_test)}')
y_pred = []
y_true = []

for sentence, true_label in zip(x_test, y_test):  # true_label은 숫자 인덱스
    result = predict_article(loaded_model, sentence)
    predicted_label_idx = result['predicted_article'] # 예측값 (숫자 인덱스)

    true_article = idx_to_article[true_label]

    print(f"Predicted: {predicted_label_idx}, True: {true_article}")

    # 숫자 인덱스 저장
    y_pred.append(predicted_label_idx)
    y_true.append(true_article)

# 성능 계산
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
recall = recall_score(y_true, y_pred, average="macro", zero_division=1)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)
try:
    roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
except ValueError:
    roc_auc = float('nan')

# 성능 출력
print("\n📊 테스트 데이터 성능 평가 결과 📊")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")
