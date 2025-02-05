import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertModel,
    ElectraTokenizer,
    ElectraModel,
    RobertaTokenizer,
    RobertaModel,
    AutoModel,
    AutoTokenizer,
    XLMRobertaTokenizer,
    XLMRobertaModel
)
#pip install kobert-transformers
from kobert_transformers import get_tokenizer

##############################################
# 1. KLUE-BERT Base: BertMLPClassifier
##############################################
def get_KLUE_BERT_model(hidden_size=256):
    model_name = "klue/bert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    class BertMLPClassifier(nn.Module):
        def __init__(self):
            super(BertMLPClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)  # 불공정(1) 확률 출력
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 사용
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    return BertMLPClassifier(), tokenizer

##############################################
# 2. KoBERT: KoBERTMLPClassifier
##############################################
def get_KoBERT_model(hidden_size=256):
    model_name = "monologg/kobert"
    tokenizer = get_tokenizer()

    class KoBERTMLPClassifier(nn.Module):
        def __init__(self):
            super(KoBERTMLPClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(model_name)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    return KoBERTMLPClassifier(), tokenizer

##############################################
# 3. KoELECTRA: KoELECTRAMLPClassifier
##############################################
def get_KoELECTRA_model(hidden_size=256):
    model_name = "monologg/koelectra-base-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)

    class KoELECTRAMLPClassifier(nn.Module):
        def __init__(self):
            super(KoELECTRAMLPClassifier, self).__init__()
            self.electra = ElectraModel.from_pretrained(model_name)
            self.fc1 = nn.Linear(self.electra.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # 첫 토큰 활용
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    return KoELECTRAMLPClassifier(), tokenizer

##############################################
# 4. KLUE-RoBERTa: KLUERobertaMLPClassifier
##############################################
def get_KLUE_Roberta_model(hidden_size=256):
    model_name = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class KLUERobertaMLPClassifier(nn.Module):
        def __init__(self):
            super(KLUERobertaMLPClassifier, self).__init__()
            self.roberta = AutoModel.from_pretrained(model_name)
            self.fc1 = nn.Linear(self.roberta.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰(또는 첫 토큰) 사용
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    return KLUERobertaMLPClassifier(), tokenizer


##############################################
# 6. KoSBERT: KoSBERTMLPClassifier
##############################################
def get_KoSBERT_model(hidden_size=256):
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class KoSBERTMLPClassifier(nn.Module):
        def __init__(self):
            super(KoSBERTMLPClassifier, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 평균 풀링 (모든 토큰의 평균) 사용
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
            x = self.fc1(pooled_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    return KoSBERTMLPClassifier(), tokenizer

##############################################
# 7. XLM-RoBERTa: XLMRobertaMLPClassifier
##############################################
def get_XLMRoberta_model(hidden_size=256):
    model_name = "xlm-roberta-base"
    # tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    class XLMRobertaMLPClassifier(nn.Module):
        def __init__(self):
            super(XLMRobertaMLPClassifier, self).__init__()
            self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
            self.fc1 = nn.Linear(self.xlm_roberta.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    return XLMRobertaMLPClassifier(), tokenizer


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
def train_model(model,optimizer,criterion,device, train_loader, val_loader,warmup_sched,reduce_sched, num_warmup_steps,save_path, model_file, epochs=10, patience=3):
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
    torch.save(model.state_dict(), model_file)