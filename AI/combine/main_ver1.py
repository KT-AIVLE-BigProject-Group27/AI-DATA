import os, sys, pandas as pd
sys.path.append(os.path.abspath("./AI/combine"))
import modularization_ver5 as mo
contract_hwp_path ='C:/Users/User/Desktop/AI-DATA/Data_Analysis/Contract/'
mo.initialize_models()
hwp_files = [ file for file in os.listdir(contract_hwp_path) if file.endswith(".pdf")]
results = {}
all_results = []
all_summary =[]

for hwp_file in hwp_files:
    results[hwp_file] = []
    hwp_path = os.path.join(contract_hwp_path, hwp_file)
    indentification_results, summary_results = mo.pipline(hwp_path)
    for summary in summary_results:
        all_summary.append({
                'article_number' : summary['article_number'],
                'article_title' :summary['article_title'],
                'summary' : summary['summary']
        })
    for result in indentification_results:
        all_results.append({
            '파일명': hwp_file,
            '조항 번호': result['contract_article_number'],
            '항 번호': result['contract_clause_number'],
            '호 번호': result['contract_subclause_number'],
            '문장': result['Sentence'],
            '불공정 여부': result['Unfair'],
            '불공정 확률': result['Unfair_percent'],
            '독소 여부': result['Toxic'],
            '독소 확률': result['Toxic_percent'],
            '어긴 법 조항 번호': result['law_article_number'],
            '어긴 법 항 번호': result['law_clause_number_law'],
            '어긴 법 호 번호': result['law_subclause_number_law'],
            '설명': result['explain']
        })
    print(f'----{hwp_file}----')
    for indentification_result in indentification_results:
        if indentification_result['Unfair'] == 1:
            print(f"문장: {indentification_result['Sentence']}")
            print(f"어긴 법 조항 번호: {indentification_result['law_article_number']}")
            print(f"어긴 법 항 번호: {indentification_result['law_clause_number_law']}")
            print(f"어긴 법 호 번호: {indentification_result['law_subclause_number_law']}")
            print(f"설명: {indentification_result['explain']}")
df = pd.DataFrame(all_results)
df_2 = pd.DataFrame(all_summary)
# # CSV 파일로 저장
csv_filename = "contract_analysis_results.csv"
df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
csv = "contract_summary_results.csv"
df_2.to_csv(csv, index=False, encoding="utf-8-sig")