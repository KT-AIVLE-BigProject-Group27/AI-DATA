import os
import sys
import re
sys.path.append(os.path.abspath("./AI/combine"))
sys.path.append(os.path.abspath("./Preprocessing/saperate_sentence_for_train"))
import modularization_ver2 as mo

symtostr = {
    "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
    "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10"
}
inverse_symtostr = {
    '1': '①', '2': '②', '3': '③', '4': '④', '5': '⑤',
     '6': '⑥', '7': '⑦', '8': '⑧', '9': '⑨', '10': '⑩'}

#contract_hwp_path ='C:/Users/User/Desktop/AI-DATA/Data_Analysis/Contract/example.hwp'
contract_hwp_path ='C:/Users/User/Desktop/AI-DATA/Data_Analysis/Contract/24년 개정 직매입 표준거래계약서(면세점)_예시(화장품).hwp'
txt = mo.hwp5txt_to_string(contract_hwp_path)
articles = mo.contract_to_articles_ver2(txt)
data_for_toxic = []
data_for_unfair = []
for article_number, article_detail in articles.items():
    print(f'*******************{article_number}조 문장 분리 시작*******************')
    match = re.match(r"(제\s?\d+조(?:의\s?\d+)?\s?)\[(.*?)\]\s?(.+)", article_detail, re.DOTALL)
    article_title = match.group(2)
    article_content = match.group(3)
    sentences = mo.article_to_sentences(article_number, article_title, article_content)
    for sentence in sentences:
        if sentence[4] != '':
            data_for_unfair.append([inverse_symtostr[sentence[3]] + ' ' + sentence[4] + ' ' + sentence[5] + '. ' + sentence[6],0,0,sentence[0]])
            data_for_toxic.append([inverse_symtostr[sentence[3]] + ' ' + sentence[4] + ' ' + sentence[5] + '. ' + sentence[6], 0,sentence[0]])
            if sentence[6] != '':
                data_for_unfair.append([inverse_symtostr[sentence[3]] + ' ' + sentence[4], 0, 0, sentence[0]])
                data_for_toxic.append([inverse_symtostr[sentence[3]] + ' ' + sentence[4], 0,sentence[0]])
        else:
            data_for_unfair.append([f'{sentence[2]} {sentence[5]}. {sentence[6]}', 0, 0, sentence[0]])
            data_for_toxic.append([f'{sentence[2]} {sentence[5]}. {sentence[6]}', 0,sentence[0]])
            if sentence[6] != '':
                data_for_unfair.append([f'{sentence[2]}', 0, 0, sentence[0]])
                data_for_toxic.append([f'{sentence[2]}', 0, sentence[0]])



save_path_u =  "./Data_Analysis/Data_ver2/unfair_data/normal_2_preprocessing.csv"
save_path_t =  "./Data_Analysis/Data_ver2/toxic_data/normal_2_preprocessing.csv"
import pandas as pd
df_u = pd.DataFrame(data_for_unfair, columns=["sentence", "unfair_label","law_article","article_number"])
df_t = pd.DataFrame(data_for_toxic, columns=["sentence", "toxic_label","article_number"])
df_u_deduplicated = df_u.drop_duplicates()
df_t_deduplicated = df_t.drop_duplicates()
df_u_deduplicated.to_csv(save_path_u, index=False, encoding="utf-8-sig")
df_t_deduplicated.to_csv(save_path_t, index=False, encoding="utf-8-sig")