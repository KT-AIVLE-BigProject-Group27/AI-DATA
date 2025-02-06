################################################################################################
# 필요 패키지 import
################################################################################################
import subprocess, pickle, openai, torch, json, os, re, fitz, numpy as np, torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, ElectraTokenizer, ElectraModel
from kobert_transformers import get_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
######################## hwp5txt path ########################
# 진석
# hwp5txt_exe_path =
# 계승
# hwp5txt_exe_path = "C:/Users/LeeGyeSeung/Desktop/KT_AIVLE/빅프로젝트폴더/KT_AIVLE_Big_Project/Data_Analysis/Contract/hwp5txt.exe"
# 명재
hwp5txt_exe_path = 'C:/Users/User/anaconda3/envs/bigp/Scripts/hwp5txt.exe'
################################################################################################
# Hwp파일에서 Text 추출
################################################################################################
def hwp5txt_to_string(hwp_path):
    if not os.path.exists(hwp_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {hwp_path}")
    command = f"{hwp5txt_exe_path} \"{hwp_path}\""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    extracted_text = result.stdout
    return extracted_text
################################################################################################
# PDF파일에서 Text 추출
################################################################################################
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    text = text.replace('\n', ' ')
    text = re.sub(r'-\s?\d+\s?-', '', text)
    return text
################################################################################################
# # 날짜 형식 변형 (예: 2023. 01. 01. => 2023-01-01)
################################################################################################
def replace_date_with_placeholder(content):
    content_with_placeholder = re.sub(r"(\d{4})\.\s(\d{2})\.\s(\d{2}).", r"\1-\2-\3", content)
    return content_with_placeholder
################################################################################################
# 전체 계약서 텍스트를 받아, 조를 분리하는 함수 ver2
################################################################################################
def contract_to_articles_ver2(txt):
    pattern, key_pattern = r'(제\d+조(?:의\d+)? \[.+?\])', r'제(\d+)조(?:의(\d+))? \[.+?\]'
    matches = list(re.finditer(pattern, txt))
    contract_sections = {}
    for i, match in enumerate(matches):
        section_title = match.group(0)
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        section_content = txt[start_idx:end_idx].strip()
        section_content = section_content.replace('“', '').replace('”', '').replace('\n', '').replace('<표>','')
        match_title = re.match(key_pattern, section_title)
        main_num = match_title.group(1)
        try:
            sub_num = match_title.group(2)
        except IndexError:
            sub_num = None
        key = f"{main_num}-{sub_num}" if sub_num else main_num
        contract_sections[key] = section_title + ' ' + section_content
    return contract_sections
################################################################################################
# 조를 받아 문장으로 분리
################################################################################################
def split_once_by_clauses(content):
    pattern = r"(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)"
    matches = list(re.finditer(pattern, content))
    result = []
    for i, match in enumerate(matches):
        end = match.end()
        if i + 1 < len(matches):
            next_start = matches[i + 1].start()
            clause_content = content[end:next_start].strip()
        else:
            clause_content = content[end:].strip()
        result.append(match.group())
        result.append(clause_content)
    return result

def split_once_by_sub_clauses(content):
    pattern = r"(\d+\.)"
    matches = list(re.finditer(pattern, content))
    result = []
    for i, match in enumerate(matches):
        end = match.end()
        if i + 1 < len(matches):
            next_start = matches[i + 1].start()
            clause_content = content[end:next_start].strip()
        else:
            clause_content = content[end:].strip()
        result.append(match.group())
        result.append(clause_content)
    return result


def article_to_sentences(article_number,article_title, article_content):
    sentences = []
    if '①' in article_content:
        clause_sections = split_once_by_clauses(article_content)
        for i in range(0, len(clause_sections), 2):
            clause_number = clause_sections[i]
            clause_content = clause_sections[i + 1]
            if '1.' in clause_content:
                sub_clause_sections = split_once_by_sub_clauses(clause_content)
                sentences.append([article_number.strip(), article_title.strip(), '', clause_number.strip(),clause_content.split('1.')[0].strip(), '', ''])
                for j in range(0, len(sub_clause_sections), 2):
                    sub_clause_number = sub_clause_sections[j]
                    sub_clause_content = sub_clause_sections[j + 1]
                    sentences.append([article_number.strip(), article_title.strip(), '', clause_number.strip(), clause_content.split('1.')[0].strip(), sub_clause_number[0].strip(), sub_clause_content.strip()])
            else:
                sentences.append([article_number.strip(), article_title.strip(), '', clause_number.strip(), clause_content.strip(), '', ''])
    elif '1.' in article_content:
        sub_clause_sections = split_once_by_sub_clauses(article_content)
        sentences.append([article_number.strip(), article_title.strip(), article_content.split('1.')[0].strip(), '', '', '', ''])
        for j in range(0, len(sub_clause_sections), 2):
            sub_clause_number = sub_clause_sections[j]
            sub_clause_content = sub_clause_sections[j + 1]
            sentences.append([article_number.strip(), article_title.strip(), article_content.split('1.')[0].strip(), '', '', sub_clause_number.strip(),sub_clause_content.strip()])
    else:
        sentences.append([article_number.strip(),article_title.strip(),article_content.strip(),'','','',''])
    return sentences
################################################################################################
# 모델 로드
################################################################################################
def load_trained_model_statice(model_class, model_file):
    model = model_class().to(device)
    state_dict = torch.load(model_file, map_location=device)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        raise TypeError(f"모델 가중치 로드 실패: {model_file} (잘못된 데이터 타입 {type(state_dict)})")

################################################################################################
# 토크나이징 모델 로드 & 전체 모델과 데이터 로드
################################################################################################
def initialize_models():
    global unfair_model, unfair_tokenizer, article_model, article_tokenizer, toxic_model, toxic_tokenizer,law_tokenizer,law_model, law_data, law_embeddings, device, summary_model, summary_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('law model loading...')
    law_tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    law_model = BertModel.from_pretrained("klue/bert-base").to(device)
    print('summary model loading...')
    summary_md_dir = 'C:/Users/User/Desktop/Model/exaone_prompt_model'
    summary_model = AutoModelForCausalLM.from_pretrained(summary_md_dir,trust_remote_code=True)
    summary_tokenizer = AutoTokenizer.from_pretrained(summary_md_dir,trust_remote_code=True)

    class KoBERTMLPClassifier(nn.Module):
        def __init__(self):
            super(KoBERTMLPClassifier, self).__init__()
            self.bert = BertModel.from_pretrained("monologg/kobert")
            self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.sigmoid(x)

    class KoELECTRAMLPClassifier(nn.Module):
        def __init__(self):
            super(KoELECTRAMLPClassifier, self).__init__()
            self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")
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

    class BertArticleClassifier(nn.Module):
        def __init__(self, bert_model_name="klue/bert-base", hidden_size=256, num_classes=10):
            super(BertArticleClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, num_classes)  # 조항 개수만큼 출력
            self.softmax = nn.Softmax(dim=1)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 벡터 사용
            x = self.fc1(cls_output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return self.softmax(x)  # 확률 분포 출력
    # 불공정 조항 판별 모델 로드
    print('unfair model loading...')
    unfair_model = load_trained_model_statice(KoBERTMLPClassifier, f"./Model/unfair_identification/KoBERT_mlp.pth")
    unfair_tokenizer = get_tokenizer()
    # 조항 예측 모델 로드
    print('article model loading...')
    article_model = load_trained_model_statice(BertArticleClassifier, f"./Model//article_prediction/klue_bert_mlp.pth")
    article_tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    # 독소 조항 판별 모델 로드
    print('toxic model loading...')
    toxic_model = load_trained_model_statice(KoELECTRAMLPClassifier, f"./Model/toxic_identification/KoELECTRA_mlp.pth")
    toxic_tokenizer =ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
    # 법률 데이터 로드
    with open("./Data_Analysis/law/law_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    with open("./Data_Analysis/law/law_data_ver2.json", "r", encoding="utf-8") as f:
        law_data = json.load(f)
    law_embeddings = np.array(data["law_embeddings"])
    print("All models and data have been successfully loaded")
################################################################################################
# 불공정 식별
################################################################################################
def predict_unfair_clause(sentence):
    unfair_model.eval()
    inputs = unfair_tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = unfair_model(inputs["input_ids"], inputs["attention_mask"])
        unfair_prob = output.item()
    return 1 if unfair_prob >= 0.4986 else 0, unfair_prob
################################################################################################
# 조항 예측
################################################################################################
def predict_article(sentence):
    idx_to_article = {0: '11', 1: '13', 2: '14', 3: '15', 4: '16', 5: '17', 6: '19', 7: '22', 8: '6', 9: '9'}
    article_model.eval()
    inputs = article_tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = article_model(inputs["input_ids"], inputs["attention_mask"])
        predicted_idx = idx_to_article[torch.argmax(output).item()]
    return predicted_idx
################################################################################################
# 독소 식별
################################################################################################
def predict_toxic_clause(sentence):
    toxic_model.eval()
    inputs = toxic_tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        output = toxic_model(inputs["input_ids"], inputs["attention_mask"])
        toxic_prob = output.item()
    return 1 if toxic_prob >= 0.4938 else 0, toxic_prob
################################################################################################
# 코사인 유사도
################################################################################################
def find_most_similar_law_within_article(sentence, predicted_article, law_data):
    contract_embedding = law_model(**law_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
    predicted_article = str(predicted_article)
    matching_article = []
    for article in law_data:
        if article["article_number"].split()[1].startswith(predicted_article):
            matching_article.append(article)
    if not matching_article:
        return {
            "Article number": None,
            "Article title": None,
            "clause number": None,
            "subclause number": None,
            "Article detail": None,
            "clause detail": None,
            "subclause detail": None,
            "similarity": None
        }
    best_match = None
    best_similarity = -1
    for article in matching_article:
        article_title = article.get("article_title", None)
        article_detail = article.get("article_content", None)
        if article_title:
            article_embedding = law_model(**law_tokenizer(article_title, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
            similarity = cosine_similarity([contract_embedding], [article_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "article_number": article["article_number"],
                    "article_title": article_title,
                    "article_detail": article_detail,
                    "clause_number": None,
                    "clause_detail": None,
                    "subclauses": None,
                    "similarity": best_similarity
                }
        for clause in article.get("clauses", []):
            clause_text = clause["content"].strip()
            clause_number = clause["clause_number"]
            if clause_text:
                clause_embedding = law_model(**law_tokenizer(clause_text, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
                similarity = cosine_similarity([contract_embedding], [clause_embedding])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "article_number": article["article_number"],
                        "article_title": article_title,
                        "article_detail": article_detail,
                        "clause_number": clause_number,
                        "clause_detail": clause_text,
                        "subclauses": clause.get("sub_clauses", []),
                        "similarity": best_similarity
                    }
        if best_match and best_match["subclauses"]:
            for subclause in best_match["subclauses"]:
                if isinstance(subclause, str):
                    subclause_embedding = law_model(**law_tokenizer(subclause, return_tensors="pt", padding=True, truncation=True).to(device)).pooler_output.cpu().detach().numpy()[0]
                    similarity = cosine_similarity([contract_embedding], [subclause_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match["subclause_detail"] = subclause
    if best_match is None:
        return {
            "Article number": f"Article {predicted_article}",
            "Article title": None,
            "clause number": None,
            "subclause number": None,
            "Article detail": None,
            "clause detail": None,
            "subclause detail": None,
            "similarity": None
        }
    return {
        "Article number": best_match["article_number"],
        "Article title": best_match["article_title"],
        "clause number": f"{best_match['clause_number']}" if best_match["clause_number"] else None,
        "subclause number": best_match.get("subclause_detail")[0] if best_match.get("subclause_detail") else None,
        "Article detail": best_match["article_detail"],
        "clause detail": best_match["clause_detail"],
        "subclause detail": best_match.get("subclause_detail", None),
        "similarity": best_similarity
    }
################################################################################################
# 설명 AI
################################################################################################
def explanation_AI(sentence, unfair_label, toxic_label, law=None):
    with open('./Key/openAI_key.txt', 'r') as file:
        openai.api_key = file.readline().strip()
    os.environ['OPENAI_API_KEY'] = openai.api_key
    client = openai.OpenAI()
    if unfair_label == 0 and toxic_label == 0:
        return None
    prompt = f"""
        아래 계약 조항이 특정 법률을 위반하는지 분석하고, 조항(제n조), 항(제m항), 호(제z호) 형식으로 **명확하고 간결하게** 설명하세요.
        📌 **설명할 때는 사용자에게 직접 말하는 듯한 자연스러운 문장으로 구성하세요.**
        📌 **한눈에 보기 쉽도록 짧고 명확한 문장을 사용하세요.**
        📌 **불공정 라벨이 1인 경우에는 불공정에 관한 설명만 하고, 독소 라벨이 1인 경우에는 독소에 관한 설명한 하세요**

        계약 조항: "{sentence}"
        불공정 라벨: {unfair_label} (1일 경우 불공정)
        독소 라벨: {toxic_label} (1일 경우 독소)   
        {f"관련 법 조항: {law}" if law else "관련 법 조항 없음"}

        🔴 **불공정 조항일 경우:**
        1️⃣ **위반된 법 조항을 '제n조 제m항 제z호' 형식으로 먼저 말해주세요.**
        2️⃣ **위반 이유를 간결하게 설명하세요.**
        3️⃣ **설명은 '🚨 법 위반!', '🔍 이유' 순서로 구성하세요.**

        ⚫ **독소 조항일 경우:**
        1️⃣ **법 위반이 아니라면, 해당 조항이 계약 당사자에게 어떤 위험을 초래하는지 설명하세요.**
        2️⃣ **구체적인 문제점을 짧고 명확한 문장으로 설명하세요.**
        3️⃣ **설명은 '💀 독소 조항', '🔍 이유' 순서로 구성하세요.**

        ⚠️ 참고: 제공된 법 조항이 실제로 위반된 조항이 아닐 경우, **GPT가 판단한 적절한 법 조항을 직접 사용하여 설명하세요.** 
        그러나 원래 제공된 법 조항과 비교하여 반박하는 방식으로 설명하지 마세요.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content":
                            "당신은 계약서 조항이 특정 법률을 위반하는지 분석하는 법률 전문가입니다. \
                            불공정 조항의 경우, 어떤 법 조항을 위반했는지 조항(제n조), 항(제m항), 호(제z호) 형식으로 정확히 명시한 후 설명하세요. \
                            만약 제공된 법 조항이 실제로 위반된 조항이 아니라면, GPT가 판단한 적절한 법 조항을 사용하여 설명하세요. \
                            독소 조항은 법률 위반이 아니라 계약 당사자에게 미치는 위험성을 중심으로 설명하세요."
                   },
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    ).choices[0].message.content
    return response

################################################################################################
# 요약 AI
################################################################################################
def article_summary_AI_ver2(article):
    # 제목 추출 (정규식 사용)
    article_title_match = re.search(r"(제\d+조의?\s?\d?\s?\[.*?\])", article)
    # 제목 추출 (매칭된 제목을 가져오되, 없으면 빈 문자열로 설정)
    article_title = article_title_match.group(0) if article_title_match else ""
    
    # 제목을 제외한 나머지 내용
    article_temp = re.sub(r"제\d+조의?\s?\d?\s?\[.*?\]", "", article)  # 제목 제거
    prompt = f"""
  원본 문장:{article_temp}
원본 문장의 맥락을 살펴보고, 빠르게 문장을 요약하여 재구성합니다.
제목은 그대로 두시고, 내용의 핵심을 추출하여 전체적으로 요약하면 됩니다.
결과는 하나의 문장으로 표현하면 됩니다.
말 끝을 번역문이 아니라 자연스러운 한글 문장이 되도록 가공합니다.
괄호 () 속 내용 보다는 문장 전체의 맥락을 더 중요하게 봅니다.
문장을 생성할 때, '다' 로 끝나게 합니다.
문장의 조금만 더 간략하게 요약합니다.
"""

    messages = [
        {"role": "system", "content": "You are an excellent sentence summarizer. You understand the context and concisely summarize key sentences as an assistant."},
        {"role": "user", "content": prompt}
    ]

    # 모델과 tokenizer가 있는 디바이스 설정 (cuda가 가능하면 cuda 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_model.to(device)

    # input_ids 생성 및 디바이스로 이동
    input_ids = summary_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # 모델에서 텍스트 생성
    output = summary_model.generate(
        input_ids,
        eos_token_id=summary_tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False
    )

    # 생성된 텍스트 디코딩
    generated_text = summary_tokenizer.decode(output[0], skip_special_tokens=True)

    # 프롬프트 길이를 기준으로 생성된 부분만 추출
    generated_text_only = generated_text[len(summary_tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    summary = generated_text_only.strip()
    summary = re.sub(r"\*\*요약 문장:\*\*:\s*", "", summary)
    summary = re.sub(r"\*\*요약 문장:\*\*:", "", summary)
    summary = re.sub(r"\*\*요약 문장:\*\*", "", summary)
    summary = re.sub("\n", "", summary)
    summary = article_title + ' ' + summary
    return summary

################################################################################################
# 파이프 라인
################################################################################################
def pipline(contract_path):
    indentification_results = []
    summary_results = []
    print('Extracting text from the Hangul file...')
    txt = hwp5txt_to_string(contract_path)
    print('Splitting text into article sections...')
    txt = replace_date_with_placeholder(txt)
    articles = contract_to_articles_ver2(txt)

    for article_number, article_detail in articles.items():
        print(f'Analyzing Article {article_number}...')
        match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
        article_title = match.group(2)
        article_content = match.group(3)
        sentences = article_to_sentences(article_number,article_title, article_content)
        # summary = article_summary_AI_ver2(article_detail)
        # summary_results.append(
        #                 {
        #                 'article_number':article_number, # 조 번호
        #                 'article_title': article_title, # 조 제목
        #                 'summary': summary # 조 요약
        #                 }
        # )
        for article_number, article_title, article_content, clause_number, clause_detail, subclause_number, subclause_detail in sentences:
            sentence = re.sub(r'\s+', ' ', f'[{article_title}] {article_content} {clause_number} {clause_detail} {subclause_number + "." if subclause_number else ""} {subclause_detail}').strip()
            unfair_result, unfair_percent = predict_unfair_clause(sentence)
            if unfair_result:
                predicted_article = predict_article(sentence)  # 예측된 조항
                law_details = find_most_similar_law_within_article(sentence, predicted_article, law_data)
                toxic_result = 0
                toxic_percent = 0
            else:
                toxic_result, toxic_percent = predict_toxic_clause(sentence)
                law_details = {
                    "Article number": None,
                    "Article title": None,
                    "clause number": None,
                    "subclause number": None,
                    "Article detail": None,
                    "clause detail": None,
                    "subclause detail": None,
                    "similarity": None
                }
            law_text = []
            if law_details.get("Article number"):
                law_text.append(f"{law_details['Article number']}({law_details['Article title']})")
            if law_details.get("Article detail"):
                law_text.append(f": {law_details['Article detail']}")
            if law_details.get("clause number"):
                law_text.append(f" {law_details['clause number']}: {law_details['clause detail']}")
            if law_details.get("subclause number"):
                law_text.append(f" {law_details['subclause number']}: {law_details['subclause detail']}")
            law = "".join(law_text) if law_text else None

            explain = explanation_AI(sentence, unfair_result, toxic_result, law)

            if unfair_result or toxic_result:
                indentification_results.append(
                                {
                                    'contract_article_number': article_number if article_number != "" else None, # 계약서 조
                                    'contract_clause_number' : clause_number if clause_number != "" else None, # 계약서 항
                                    'contract_subclause_number': subclause_number if subclause_number != "" else None, # 계약서 호
                                    'Sentence': sentence, # 식별
                                    'Unfair': unfair_result, # 불공정 여부
                                    'Unfair_percent': unfair_percent, # 불공정 확률
                                    'Toxic': toxic_result,  # 독소 여부
                                    'Toxic_percent': toxic_percent,  # 독소 확률
                                    'law_article_number': law_details['Article number'],  # 어긴 법 조   (불공정 1일때, 아니면 None)
                                    'law_clause_number_law': law_details['clause number'], # 어긴 법 항 (불공정 1일때, 아니면 None)
                                    'law_subclause_number_law': law_details['subclause number'],  # 어긴 법 호 (불공정 1일때, 아니면 None)
                                    'explain': explain #explain (불공정 1또는 독소 1일때, 아니면 None)
                                    }
                )
    return indentification_results, summary_results
    ############################################################################################################
    ############################################################################################################
