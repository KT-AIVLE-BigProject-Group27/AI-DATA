import os
import sys
import time
import re

sys.path.append(os.path.abspath("./AI/combine"))
import modularization_ver2 as mo

# 계약서 이름 설정
contract_hwp_path ='C:/Users/User/Desktop/bigp/AI-DATA/Data_Analysis/Contract/example.hwp'
contract_pdf_path ='C:/Users/User/Desktop/bigp/AI-DATA/Data_Analysis/Contract/example.pdf'
# 모델 초기화


# 실행 시간 측정
start_time = time.time()  # 시작 시간 기록
indentification_results = []
summary_results = []
print('텍스트 추출')
#txt = mo.hwp5txt_to_string(contract_hwp_path)
txt = mo.extract_text_from_pdf(contract_pdf_path)
print('텍스트를 조 단위로 분리')
articles = mo.contract_to_articles_ver2(txt)
for article_number, article_detail in articles.items():
    print(f'*******************{article_number}조 문장 분리 시작*******************')
    match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
    article_title = match.group(2)
    article_content = match.group(3)
    sentences = mo.article_to_sentences(article_number, article_title, article_content)
    for sentence in sentences:
        print(sentence)
    # summary = article_summary_AI_ver2(prompt, article_detail)
    # summary_results.append(
    #     {
    #         'article_number': article_number,  # 조 번호
    #         'article_title': article_title,  # 조 제목
    #         'summary': summary  # 조 요약
    #     }
    # )
    # print(f'{article_number}조 요약: {summary}')
    # pre_article_detail = ''
    # pre_clause_detail = ''
    # pre_subclause_detail = ''
    # for _, _, _, clause_number, clause_detail, subclause_number, subclause_detail in sentences:
    #     for idx, sentence in enumerate([article_detail, clause_detail, subclause_detail]):
    #         if idx == 0:
    #             if pre_article_detail == sentence:
    #                 continue
    #             else:
    #                 pre_article_detail = sentence
    #         elif idx == 1:
    #             if pre_clause_detail == sentence:
    #                 continue
    #             else:
    #                 pre_clause_detail = sentence
    #
    #         elif idx == 2:
    #             if pre_subclause_detail == sentence:
    #                 continue
    #             else:
    #                 pre_subclause_detail = sentence
    #
    #         # print(f'sentence: {sentence}')
    #         unfair_result, unfair_percent = mo.predict_unfair_clause(unfair_model, sentence, 0.5011)
    #         if unfair_result:
    #             # print('불공정!!!')
    #             predicted_article = mo.predict_article(article_model, sentence)  # 예측된 조항
    #             law_details = mo.find_most_similar_law_within_article(sentence, predicted_article, law_data)
    #             toxic_result = 0
    #             toxic_percent = 0
    #         else:
    #             toxic_result, toxic_percent = mo.predict_toxic_clause(toxic_model, sentence, 0.5011)
    #             # print('독소!!!' if toxic_result else '일반!!!')
    #             law_details = {
    #                 "Article number": None,
    #                 "Article title": None,
    #                 "Paragraph number": None,
    #                 "Subparagraph number": None,
    #                 "Article detail": None,
    #                 "Paragraph detail": None,
    #                 "Subparagraph detail": None,
    #                 "similarity": None
    #             }
    #         law_text = []
    #         if law_details.get("Article number"):
    #             law_text.append(f"{law_details['Article number']}({law_details['Article title']})")
    #         if law_details.get("Article detail"):
    #             law_text.append(f": {law_details['Article detail']}")
    #         if law_details.get("Paragraph number"):
    #             law_text.append(f" {law_details['Paragraph number']}: {law_details['Paragraph detail']}")
    #         if law_details.get("Subparagraph number"):
    #             law_text.append(f" {law_details['Subparagraph number']}: {law_details['Subparagraph detail']}")
    #         law = "".join(law_text) if law_text else None
    #
    #         # explain = explanation_AI(sentence, unfair_result, toxic_result, law)
    #
    #         if unfair_result or toxic_result:
    #             indentification_results.append(
    #                 {
    #                     'contract_article_number': article_number if article_number != "" else None,  # 계약서 조
    #                     'contract_clause_number': clause_number if clause_number != "" else None,  # 계약서 항
    #                     'contract_subclause_number': subclause_number if subclause_number != "" else None,  # 계약서 호
    #                     'Sentence': sentence,  # 식별
    #                     'Unfair': unfair_result,  # 불공정 여부
    #                     'Unfair_percent': unfair_percent,  # 불공정 확률
    #                     'Toxic': toxic_result,  # 독소 여부
    #                     'Toxic_percent': toxic_percent,  # 독소 확률
    #                     'law_article_number': law_details['Article number'],  # 어긴 법 조   (불공정 1일때, 아니면 None)
    #                     'law_clause_number_law': law_details['Paragraph number'],  # 어긴 법 항 (불공정 1일때, 아니면 None)
    #                     'law_subclause_number_law': law_details['Subparagraph number'],  # 어긴 법 호 (불공정 1일때, 아니면 None)
    #                     'explain': None  # explain (불공정 1또는 독소 1일때, 아니면 None)
    #                 }
    #             )
end_time = time.time()  # 종료 시간 기록

# 실행 시간 출력
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
