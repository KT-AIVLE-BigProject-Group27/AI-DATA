from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import os, re, sys
sys.path.append(os.path.abspath("../combine"))
import modularization_ver1 as mo

def article_summary_AI(model, tokenizer, prompt, input_text, max_length=256):
    input_ids = tokenizer.encode(f"{prompt}{input_text}", return_tensors="pt")
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 요약 프롬프트 설정
prompt = (
    """
다음은 계약서의 조항입니다. 이 조항의 핵심 내용을 아래 기준에 따라 간결하게 요약하세요:

1. **조항 제목**: 조항 제목을 명확히 명시하세요.
2. **주요 목적**: 이 조항이 규정하는 목적 또는 대상을 한 문장으로 요약하세요.
3. **권리와 의무**: 갑과 을의 권리와 의무를 각각 구분하여 간략히 기술하세요.
4. **이행 절차**: 이 조항에서 요구하는 이행 절차나 조건이 있으면 간단히 정리하세요.
5. **위반 시 조치**: 위반 시 발생하는 결과나 책임을 간결하게 요약하세요.

요약은 **조항 제목**으로 시작하며, 각 항목을 **짧고 명확한 문장**으로 작성하세요. 불필요한 반복을 피하고, 법률 용어는 일반인이 이해하기 쉽게 풀이하세요.

예시:
"제4조 [상품의 납품]: 을은 갑에게 상품을 정해진 기한과 장소에 납품해야 하며, 납품 지연 시 사전 승인 요청이 필요하다. 갑은 검수 후 하자가 없으면 상품을 수령해야 한다."
"""
)
# 구글 다국어 mT5 모델 로드
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

# 계약서 텍스트 변환 및 조항 추출
summary_results = []

# 파일 경로 설정
file_path = 'example(pdf_to_txt).txt'

# 파일 읽기 모드로 열기
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

articles = mo.contract_to_articles_ver2(text)

for article_number, article_detail in articles.items():
    print('*'*50)
    match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
    article_title = match.group(2)
    article_content = match.group(3)

    summary = article_summary_AI(model, tokenizer,prompt, article_detail)
    summary_results.append(
        {
            'article_number': article_number,  # 조 번호
            'article_title': article_title,  # 조 제목
            'summary': summary  # 조 요약
        }
    )
    print(f'{article_number}조 요약: {summary}')



# 모델 저장 경로 설정
save_directory = "C:/Model"

# 모델 및 토크나이저 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

