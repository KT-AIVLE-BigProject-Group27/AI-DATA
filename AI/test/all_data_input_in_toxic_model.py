import torch
import torch.nn as nn
import pandas as pd
import os
from transformers import (
    ElectraTokenizer,
    ElectraModel,
)
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
# ✅ KLUE/BERT 토크나이저 및 모델 로드

model_name = "monologg/koelectra-base-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_name)

# model_name = "klue/bert-base"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# ###############################################################################################################################################
# 독소 데이터 로드 및 전처리
###############################################################################################################################################
directory_path = './Data_Analysis/Data_ver2/toxic_data/'
files_to_merge = [f for f in os.listdir(directory_path) if 'preprocessing' in f and f.endswith('.csv')]
merged_df_toxic = pd.DataFrame()
for file in files_to_merge:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)
    df = df[['sentence','toxic_label','article_number']]
    merged_df_toxic = pd.concat([merged_df_toxic, df], ignore_index=True)
merged_df_toxic["article_number"] = merged_df_toxic["article_number"].astype(str)
merged_df_toxic["sentence"] = merged_df_toxic.apply(
    lambda row: f"{article_to_title.get(row['article_number'])} {row['sentence']}", axis=1
)
merged_df_toxic["unfair_label"] = 0
###############################################################################################################################################
# 위반 데이터 로드 및 전처리
###############################################################################################################################################
directory_path = './Data_Analysis/Data_ver2/unfair_data/'
files_to_merge = [f for f in os.listdir(directory_path) if 'preprocessing' in f and f.endswith('.csv')]
merged_df_unfair = pd.DataFrame()
for file in files_to_merge:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)
    df = df[['sentence','unfair_label','article_number']]
    merged_df_unfair = pd.concat([merged_df_unfair, df], ignore_index=True)
merged_df_unfair["article_number"] = merged_df_unfair["article_number"].astype(str)
merged_df_unfair["sentence"] = merged_df_unfair.apply(
    lambda row: f"{article_to_title.get(row['article_number'])} {row['sentence']}", axis=1
)
merged_df_unfair["toxic_label"] = 0
merged_df_unfair = merged_df_unfair.loc[merged_df_unfair['unfair_label']==0]
###############################################################################################################################################
# 데이터 병합 및 중복제거
###############################################################################################################################################
merged_df = pd.concat([merged_df_toxic, merged_df_unfair], ignore_index=True)
merged_df = merged_df.drop_duplicates()
print(len(merged_df))
###############################################################################################################################################
# 위반 모델 로드 및 test
###############################################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_file = "E:/Model/ver2/toxic_identification_ver3_비교/KoELECTRA/KoELECTRA_mlp.pth"

class KoELECTRAMLPClassifier(nn.Module):
    def __init__(self):
        super(KoELECTRAMLPClassifier, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.electra.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 첫 토큰 활용
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def predict_toxic_clause(c_model, sentence):
    c_model.eval()
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = c_model(inputs["input_ids"], inputs["attention_mask"])
        toxic_prob = output.item()
    return {
        "toxic_probability": round(toxic_prob * 100, 2),
    }

def load_trained_model(model_file):
    model = KoELECTRAMLPClassifier().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(f"✅ 저장된 모델 로드 완료: {model_file}")
    return model
loaded_model = load_trained_model(model_file)

results = []
for index, row in merged_df.iterrows():
    sentence, toxic_label, article_number, unfair_label = row['sentence'], row['toxic_label'], row['article_number'], row['unfair_label']
    result = predict_toxic_clause(loaded_model,sentence)
    results.append([sentence,unfair_label,toxic_label,result['toxic_probability'],article_number])


columns = ["sentence", "unfair_label", "toxic_label", "toxic_probability", "article_number"]
df = pd.DataFrame(results, columns=columns)


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 폰트 경로 설정 (맑은고딕)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 기본 한글 폰트

# 폰트 적용
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

# 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

print(f"✅ 한글 폰트 적용 완료: {font_prop.get_name()}")

# 데이터 필터링: 위반 조항, 독소 조항, 일반 조항 분리
df_toxic = df[df['toxic_label'] == 1]  # 독소 조항
df_normal = df[(df['unfair_label'] == 0) & (df['toxic_label'] == 0)]  # 일반 조항
print(f'toxic: {len(df_toxic)}')
print(f'normal: {len(df_normal)}')

# 위반 확률 분포 시각화 (히스토그램)
plt.figure(figsize=(12, 6))
sns.histplot(df_toxic['toxic_probability'], bins=20, label='독소 조항', color='blue', alpha=0.6)
sns.histplot(df_normal['toxic_probability'], bins=20, label='일반 조항', color='green', alpha=0.6)
plt.xlabel('독소 확률 (%)')
plt.ylabel('조항 개수')
plt.title('독소 확률 분포 (독소/일반 조항)')
plt.legend()
plt.savefig('./AI/test/toxic_histogram.png')
plt.close()
# 위반 확률 분포 시각화 (히스토그램)
plt.figure(figsize=(12, 6))
sns.histplot(df_toxic['toxic_probability'], bins=20, label='독소 조항', color='blue', alpha=0.6)
sns.histplot(df_normal['toxic_probability'], bins=20, label='일반 조항', color='green', alpha=0.6)
plt.xlabel('독소 확률 (%)')
plt.ylabel('조항 개수')
plt.title('독소 확률 분포 (독소/일반 조항)')
plt.legend()
plt.ylim(0, 50)
plt.savefig('./AI/test/toxic_histogram_2.png')
plt.close()
# 커널 밀도 추정 (KDE Plot)
plt.figure(figsize=(10, 6))
sns.kdeplot(df_toxic['toxic_probability'], label="독소 조항", color="blue", fill=True, alpha=0.3)
sns.kdeplot(df_normal['toxic_probability'], label="일반 조항", color="green", fill=True, alpha=0.3)
plt.xlabel('독소 확률 (%)')
plt.ylabel('조항 개수')
plt.title('독소 확률 분포 (KDE plot)')
plt.legend()
plt.savefig('./AI/test/toxic_KED.png')
plt.close()

# 위반 확률 분포 시각화 (박스플롯)
plt.figure(figsize=(8, 6))
sns.boxplot(data=[df_toxic['toxic_probability'], df_normal['toxic_probability']],
            palette=["blue", "green"])
plt.xticks(ticks=[0, 1], labels=["독소 조항", "일반 조항"])
plt.xlabel('독소 확률 (%)')
plt.title("독소 확률 분포 (박스플롯)")
plt.savefig('./AI/test/toxic_boxplot.png')
plt.close()


print(df_toxic[['sentence','toxic_probability']].sort_values('toxic_probability',ascending=False))



#######################################################################################################################################
import sys
sys.path.append(os.path.abspath("./AI"))
from threshold_settings import find_all_thresholds
from sklearn.metrics import accuracy_score
thresholds = find_all_thresholds(loaded_model,tokenizer ,'toxic', use_train=False, device="cuda")
print("Computed thresholds:", thresholds)

# 여러 임계값을 적용하여 정확도 비교
accuracy_results = {}

for name, threshold in thresholds.items():
    # 각 임계값을 기준으로 위반(불공정) 예측
    df[f'predicted_toxic_{name}'] = df["toxic_probability"].apply(lambda x: 1 if x >= threshold else 0)

    # 정확도 계산
    accuracy = accuracy_score(df["toxic_label"], df[f'predicted_toxic_{name}'])
    accuracy_results[name] = accuracy

# 최적의 임계값 선택 (정확도가 가장 높은 값)
best_threshold = max(accuracy_results, key=accuracy_results.get)
best_accuracy = accuracy_results[best_threshold]

# 결과 출력
print(f"✅ 최적의 임계값: {best_threshold} ({thresholds[best_threshold]:.4f})")
print(f"✅ 최적 임계값 적용 시 정확도: {best_accuracy:.4f}")
