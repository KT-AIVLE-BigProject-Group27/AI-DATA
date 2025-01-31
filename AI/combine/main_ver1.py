import os
import sys
import time
import re

sys.path.append(os.path.abspath("./AI/combine"))
import modularization_ver2 as mo

# 계약서 이름 설정
contract_hwp_path ='C:/Users/User/Desktop/AI-DATA/Data_Analysis/Contract/example.hwp'
# contract_pdf_path ='C:/Users/User/Desktop/AI-DATA/Data_Analysis/Contract/example.pdf'


mo.initialize_models()
start_time = time.time()
indentification_results, summary_results = mo.pipline(contract_hwp_path)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")


for result in indentification_results:
    print(f"계약 조항 번호: {result['contract_article_number']}")
    print(f"조항 번호: {result['contract_clause_number']}")
    print(f"세부 조항 번호: {result['contract_subclause_number']}")
    print(f"문장: {result['Sentence']}")
    print(f"불공정 조항 여부: {result['Unfair']} (확률: {result['Unfair_percent']:.4f})")
    print(f"독소 조항 여부: {result['Toxic']} (확률: {result['Toxic_percent']:.4f})")
    print(f"어긴 법 조항 번호: {result['law_article_number']}")
    print(f"어긴 법 항 번호: {result['law_clause_number_law']}")
    print(f"어긴 법 호 번호: {result['law_subclause_number_law']}")
    print(f"설명: {result['explain']}")
    print("=" * 80)  # 구분선 추가

#####################################TEST#################################
# txt = mo.hwp5txt_to_string(contract_hwp_path)
# txt = mo.replace_date_with_placeholder(txt)
# articles = mo.contract_to_articles_ver2(txt)
# for article_number, article_detail in articles.items():
#     print(f'*******************{article_number}조 문장 분리 시작*******************')
#     match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
#     article_title = match.group(2)
#     article_content = match.group(3)
#     sentences = mo.article_to_sentences(article_number, article_title, article_content)
#     for article_number, article_title, article_content, clause_number, clause_detail, subclause_number, subclause_detail in sentences:
#         sentence = re.sub(r'\s+', ' ',f'[{article_title}] {article_content} {clause_number} {clause_detail} {subclause_number + "." if subclause_number else ""} {subclause_detail}').strip()
#         print(sentence)