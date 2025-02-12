################################################################################################
# 필요 패키지 import
################################################################################################
import pickle, openai, torch, json, os, re, fitz, numpy as np, torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, ElectraTokenizer, ElectraModel, BertTokenizer, BertModel
from kobert_transformers import get_tokenizer

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
    summary_model = BartForConditionalGeneration.from_pretrained('./Model/article_summary/')
    summary_tokenizer = PreTrainedTokenizerFast.from_pretrained('./Model/article_summary/')

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

    if unfair_label == 1:
        prompt = f"""
            아래 계약 조항이 특정 법률을 위반하는지 분석하고, 해당 법 조항(제n조 제m항 제z호)을 위반했다는 사실을 명확하게 설명하세요.

            계약 조항: "{sentence}"
            관련 법 조항: {law if law else "관련 법 조항 없음"}

            설명을 다음 형식으로 작성하세요:
            "어떤 법의 n조 m항 z호를 위반했습니다. 이유~~~"

            ⚠️ 제공된 법 조항이 실제로 위반된 조항이 아닐 경우, GPT가 판단한 적절한 법 조항을 직접 사용하여 설명하세요.
        """
    elif toxic_label == 1:
        prompt = f"""
            아래 계약 조항이 독소 조항인지 분석하고, 독소 조항이라면 그 이유를 설명하세요.

            계약 조항: "{sentence}"

            설명을 다음 형식으로 작성하세요:
            "무엇무엇 때문에 독소입니다."
        """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "당신은 계약서 조항이 특정 법률을 위반하는지 분석하는 법률 전문가입니다. \n불공정 조항의 경우, 어떤 법 조항을 위반했는지 조항(제n조 제m항 제z호) 형식으로 정확히 명시한 후 설명하세요. \n제공된 법 조항이 실제로 위반된 조항이 아닐 경우, GPT가 판단한 적절한 법 조항을 사용하세요. \n독소 조항은 법률 위반이 아니라 계약 당사자에게 미치는 위험성을 중심으로 설명하세요.\n 반드시 200 token 이하로 작성해주세요."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    ).choices[0].message.content

    return response.strip()
################################################################################################
# 요약 AI
################################################################################################
def article_summary_AI_ver2(article):
    prompt = (
        "다음은 계약서의 조항입니다. 이 조항의 주요 내용을 다음 기준에 따라 간략히 요약하세요:\n"
        "1. 이 조항이 규정하는 주요 목적 또는 대상\n"
        "2. 갑과 을의 권리와 의무\n"
        "3. 이행해야 할 절차와 조건\n"
        "4. 위반 시 결과 또는 조치\n\n"
        "요약은 각 기준에 따라 간결하고 명확하게 작성하며, 중복을 피하세요. "
        "조 제목과 관련된 핵심 정보를 반드시 포함하세요.\n\n"
    )
    input_ids = summary_tokenizer(f"{prompt}{article}", return_tensors="pt").input_ids
    summary_ids = summary_model.generate(input_ids, max_length=1024, num_beams=4, early_stopping=True)
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
################################################################################################
# 파이프 라인
################################################################################################
def pipline(contract_path):
    indentification_results = []
    summary_results = []
    print('Extracting text from the PDF file...')
    txt = extract_text_from_pdf(contract_path)
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
