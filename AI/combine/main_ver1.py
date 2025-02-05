import os, sys
sys.path.append(os.path.abspath("./AI/combine"))
import modularization_ver3 as mo
contract_hwp_path ='C:/Users/User/Desktop/AI-DATA/Data_Analysis/Contract/'
mo.initialize_models()
hwp_files = [ file for file in os.listdir(contract_hwp_path) if file.endswith(".hwp")]
results = {}
for hwp_file in hwp_files:
    results[hwp_file] = []
    hwp_path = os.path.join(contract_hwp_path, hwp_file)
    indentification_results, summary_results = mo.pipline(hwp_path)
    for indentification_result in indentification_results:
        if indentification_result['Unfair'] == 1:
            print(f"문장: {indentification_result['Sentence']}")
            print(f"어긴 법 조항 번호: {indentification_result['law_article_number']}")
            print(f"어긴 법 항 번호: {indentification_result['law_clause_number_law']}")
            print(f"어긴 법 호 번호: {indentification_result['law_subclause_number_law']}")
            print(f"설명: {indentification_result['explain']}")
        results[hwp_file].append([f"{indentification_result['contract_article_number']}조 {indentification_result['contract_clause_number']}항 {indentification_result['contract_subclause_number']}조 위반:{indentification_result['Unfair']} 독소:{indentification_result['Toxic']}"])

for contract_name, result in results.items():
    print('*'*20)
    print(f'계약서 종류: {contract_name}')
    for r in result:
        print(r)
#####################################TEST#################################
# for result in indentification_results:
#     print(f"계약 조항 번호: {result['contract_article_number']}")
#     print(f"조항 번호: {result['contract_clause_number']}")
#     print(f"세부 조항 번호: {result['contract_subclause_number']}")
#     # print(f"문장: {result['Sentence']}")
#     print(f"불공정 조항 여부: {result['Unfair']} (확률: {result['Unfair_percent']:.4f})")
#     print(f"독소 조항 여부: {result['Toxic']} (확률: {result['Toxic_percent']:.4f})")
#     # print(f"어긴 법 조항 번호: {result['law_article_number']}")
#     # print(f"어긴 법 항 번호: {result['law_clause_number_law']}")
#     # print(f"어긴 법 호 번호: {result['law_subclause_number_law']}")
#     # print(f"설명: {result['explain']}")
#     print("=" * 80)  # 구분선 추가

#####################################TEST#################################
# import re
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