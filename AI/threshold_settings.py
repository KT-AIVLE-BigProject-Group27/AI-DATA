from sklearn.cluster import KMeans
import numpy as np
import torch, os, sys
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.abspath("./AI"))
from identification_models import (
    get_KLUE_BERT_model,
    get_KoBERT_model,
    get_KoELECTRA_model,
    get_KLUE_Roberta_model,
    get_KoSBERT_model,
    get_XLMRoberta_model,
    train_model
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_threshold(model, train_loader=None, val_loader=None, use_train=False, device="cuda"):
    model.eval()
    y_probs, y_true = np.array([]), np.array([])
    def collect_predictions(loader):
        with torch.no_grad():
            for X_batch, mask_batch, y_batch in loader:
                X_batch, mask_batch, y_batch = X_batch.to(device), mask_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, mask_batch)
                y_probs.extend(outputs.cpu().numpy().flatten())  # 확률값 저장
                y_true.extend(y_batch.cpu().numpy().flatten())  # 실제 레이블 저장

    # ✅ Train 데이터 포함 여부 선택
    if use_train and train_loader:
        collect_predictions(train_loader)
    if val_loader:
        collect_predictions(val_loader)

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(y_probs.reshape(-1, 1))
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())  # 군집 중심 정렬
    optimal_threshold = np.mean(cluster_centers)  # 두 군집 중심의 중간값

    print(f"📌 최적 임계값: {optimal_threshold:.4f}")
    return optimal_threshold

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

# (Optional) 공통: 예측 결과 수집 함수
def collect_predictions(model, loader, device):
    """
    주어진 데이터 로더에서 모델의 예측 확률과 실제 레이블을 수집합니다.
    """
    model.eval()
    y_probs, y_true = np.array([]), np.array([])  # 빈 NumPy 배열로 초기화


    with torch.no_grad():
        for X_batch, mask_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch, mask_batch)
            probs = outputs.cpu().numpy().flatten()
            labels = y_batch.cpu().numpy().flatten()

            y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
            y_true = np.concatenate([y_true, labels]) if y_true.size else labels

    return y_probs, y_true


#####################################################################
# 1. KMeans 기반 임계값
def find_threshold_kmeans(model, train_loader=None, val_loader=None, use_train=False, device="cuda"):
    """
    KMeans를 이용하여 예측 확률의 두 클러스터 중심의 평균을 임계값으로 설정합니다.
    """
    y_probs, y_true = np.array([]), np.array([])
    if use_train and train_loader is not None:
        probs, labels = collect_predictions(model, train_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels
    if val_loader is not None:
        probs, labels = collect_predictions(model, val_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(y_probs.reshape(-1, 1))
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    optimal_threshold = np.mean(cluster_centers)
    print(f"[KMeans] Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold

#####################################################################
# 2. ROC Curve 기반 임계값 (Youden's J 통계량)
def find_threshold_roc(model, train_loader=None, val_loader=None, use_train=False, device="cuda"):
    """
    ROC Curve를 이용하여 Youden's J (tpr - fpr)가 최대가 되는 임계값을 선택합니다.
    """
    y_probs, y_true = np.array([]), np.array([])
    if use_train and train_loader is not None:
        probs, labels = collect_predictions(model, train_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels
    if val_loader is not None:
        probs, labels = collect_predictions(model, val_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]
    print(f"[ROC] Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold

#####################################################################
# 3. Precision-Recall Curve 기반 (F1 Score 최대화)
def find_threshold_pr(model, train_loader=None, val_loader=None, use_train=False, device="cuda"):
    """
    Precision-Recall Curve를 기반으로 F1 Score가 최대가 되는 임계값을 선택합니다.
    """
    y_probs, y_true = np.array([]), np.array([])
    if use_train and train_loader is not None:
        probs, labels = collect_predictions(model, train_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels
    if val_loader is not None:
        probs, labels = collect_predictions(model, val_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels

    precisions, recalls, thresh = precision_recall_curve(y_true, y_probs)
    # 계산 시 division by zero 방지를 위해 아주 작은 값을 추가
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    # precision_recall_curve의 thresh 길이는 len(precisions)-1이므로 주의
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx >= len(thresh):
        optimal_threshold = thresh[-1]
    else:
        optimal_threshold = thresh[optimal_idx]
    print(f"[PR-F1] Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold

#####################################################################
# 4. Grid Search 기반 (F1 Score 최대화)
def find_threshold_grid_f1(model, train_loader=None, val_loader=None, use_train=False, device="cuda", num_candidates=100):
    """
    후보 임계값들을 grid search하여 F1 Score가 최대가 되는 임계값을 선택합니다.
    """
    y_probs, y_true = np.array([]), np.array([])
    if use_train and train_loader is not None:
        probs, labels = collect_predictions(model, train_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels
    if val_loader is not None:
        probs, labels = collect_predictions(model, val_loader, device)
        y_probs = np.concatenate([y_probs, probs]) if y_probs.size else probs
        y_true = np.concatenate([y_true, labels]) if y_true.size else labels

    candidates = np.linspace(0, 1, num_candidates)
    best_threshold = 0.5
    best_f1 = 0.0
    for t in candidates:
        y_pred = (y_probs >= t).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t
    print(f"[Grid-F1] Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold

#####################################################################
# 5. 고정 임계값
def find_threshold_fixed(model, train_loader=None, val_loader=None, use_train=False, device="cuda", fixed_value=0.5):
    """
    고정 임계값을 반환합니다.
    """
    print(f"[Fixed] Threshold: {fixed_value:.4f}")
    return fixed_value

def load_data(data_kind,tokenizer):
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
    directory_path = f'./Data_Analysis/Data_ver2/toxic_data/'
    files_to_merge = [f for f in os.listdir(directory_path) if 'preprocessing' in f and f.endswith('.csv')]
    merged_df_toxic = pd.DataFrame()
    for file in files_to_merge:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        df = df[['sentence', 'toxic_label', 'article_number']]
        merged_df_toxic = pd.concat([merged_df_toxic, df], ignore_index=True)
    merged_df_toxic["article_number"] = merged_df_toxic["article_number"].astype(str)
    merged_df_toxic["unfair_label"] = 0
    directory_path = './Data_Analysis/Data_ver2/unfair_data/'
    files_to_merge = [f for f in os.listdir(directory_path) if 'preprocessing' in f and f.endswith('.csv')]
    merged_df_unfair = pd.DataFrame()
    for file in files_to_merge:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        df = df[['sentence', 'unfair_label', 'article_number']]
        merged_df_unfair = pd.concat([merged_df_unfair, df], ignore_index=True)
    merged_df_unfair["article_number"] = merged_df_unfair["article_number"].astype(str)
    merged_df_unfair["toxic_label"] = 0
    merged_df = pd.concat([merged_df_toxic, merged_df_unfair],ignore_index=True)
    merged_df = merged_df.drop_duplicates()
    print(f'Total merged_df: {len(merged_df)}')

    if data_kind == 'toxic':
        merged_df = merged_df.loc[merged_df['unfair_label']==0]
        merged_df = merged_df[['sentence', 'toxic_label', 'article_number']]
        merged_df.rename(columns={'toxic_label': 'label'}, inplace=True)

    elif data_kind == 'unfair':
        merged_df = merged_df[['sentence', 'unfair_label', 'article_number']]
        merged_df.rename(columns={'unfair_label': 'label'}, inplace=True)

    # article별 샘플 수 4개 미만인 경우 복제
    article_counts = merged_df["article_number"].value_counts()
    for article, count in article_counts.items():
        if count < 4:
            sample_to_duplicate = merged_df[merged_df["article_number"] == article]
            num_copies = 4 - count
            merged_df = pd.concat([merged_df] + [sample_to_duplicate] * num_copies, ignore_index=True)

    # sentence 생성: [조 제목] + 실제 문장
    merged_df["sentence"] = merged_df.apply(
        lambda row: f"{article_to_title.get(row['article_number'])} {row['sentence']}", axis=1
    )

    # Train/Test 분할 (stratify: article_number, unfair_label)
    x_temp, _, y_temp, _ = train_test_split(
        merged_df["sentence"].tolist(),
        merged_df[["label", "article_number"]],
        test_size=0.1,
        random_state=42,
        stratify=merged_df[["article_number", "label"]],
        shuffle=True
    )
    y_temp_labels = y_temp["label"]

    # Train/Validation 분할 (8:2)
    X_train, X_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_temp[["article_number", "label"]],
        shuffle=True
    )

    # 데이터 토큰화: 각 모델별 tokenizer 사용 (Train/Val)
    X_train_ids, X_train_mask = tokenize_data(X_train, tokenizer)
    X_val_ids, X_val_mask = tokenize_data(X_val, tokenizer)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train_tensor)
    val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader

#####################################################################
# 6. 모든 임계값을 반환하는 종합 함수
def find_all_thresholds(model,tokenizer ,data_kind, use_train=False, device="cuda"):
    train_loader,val_loader = load_data(data_kind,tokenizer)
    """
    여러 임계값 설정 방법으로 계산된 임계값들을 딕셔너리 형태로 반환합니다.
    """
    thresholds = {}
    thresholds['kmeans'] = find_threshold_kmeans(model, train_loader, val_loader, use_train, device)
    thresholds['roc'] = find_threshold_roc(model, train_loader, val_loader, use_train, device)
    thresholds['pr'] = find_threshold_pr(model, train_loader, val_loader, use_train, device)
    thresholds['grid_f1'] = find_threshold_grid_f1(model, train_loader, val_loader, use_train, device)
    thresholds['fixed'] = find_threshold_fixed(model, train_loader, val_loader, use_train, device)
    return thresholds

def tokenize_data(sentences, tokenizer, max_length=256):
    encoding = tokenizer(
        sentences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encoding["input_ids"], encoding["attention_mask"]