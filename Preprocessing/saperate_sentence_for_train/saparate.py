import re, pandas as pd



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


def article_to_sentences(article_number, article_content):
    article_content = article_content.replace('“','').replace('”','').replace('”','').replace('\n', '').replace('<표>','')
    symtostr = {
        "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
        "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10"
    }
    sentences = []
    if '①' in article_content:
        clause_sections = split_once_by_clauses(article_content)
        for i in range(0, len(clause_sections), 2):
            clause_number = clause_sections[i].strip()
            clause_content = clause_sections[i + 1].strip()
            sentences.append([article_number, int(symtostr[clause_number].strip()), 0, clause_number + ' ' + clause_content.split('1.')[0]])
            if '1.' in clause_content:
                sub_clause_sections = split_once_by_sub_clauses(clause_content)
                for j in range(0, len(sub_clause_sections), 2):
                    sub_clause_number = sub_clause_sections[j].strip()
                    sub_clause_content = sub_clause_sections[j+1].strip()
                    sentences.append([article_number, int(symtostr[clause_number]), int(sub_clause_number.split('.')[0]), clause_number + ' ' + clause_content.split('1.')[0] + sub_clause_number + ' '+ sub_clause_content])
    elif '1.' in article_content:
        sub_clause_sections = split_once_by_sub_clauses(article_content)
        sentences.append([article_number, 0, 0, article_content.strip()])
        for j in range(0, len(sub_clause_sections), 2):
            sub_clause_number = sub_clause_sections[j].strip()
            sub_clause_content = sub_clause_sections[j + 1].strip()
            sentences.append([article_number, 0, int(sub_clause_number.split('.')[0]), article_content.split('1.')[0] + ' ' + sub_clause_content])
    else:
        sentences.append([article_number, 0, 0, article_content.strip()])
    return sentences

def preprocessing(data_path,save_path):
    raw_data = pd.read_csv(data_path)
    data = []
    tmt = ''
    for index, row in raw_data.iterrows():
        if tmt != row['content']:
            sentences = article_to_sentences(row['article_number'], row['content'])
            tmt = row['content']

        for sentence in sentences:
            if row['sub_clause_number'] ==0:
                unfair_label = row['unfair_label']
                law_article = row['law_article']
                article_number = row['article_number']
                if [row['article_number'], row['clause_number'], row['sub_clause_number']] == sentence[0:3]:
                    data.append([sentence[3], row['unfair_label'], row['law_article'],row['article_number']])
                    break
            else:
                if [row['article_number'], row['clause_number'], row['sub_clause_number']] == sentence[0:3]:
                    data.append([sentence[3], unfair_label, law_article, article_number])
                    break

    df = pd.DataFrame(data, columns=["sentence", "unfair_label", "law_article","article_number"])
    df_deduplicated = df.drop_duplicates()
    df_deduplicated.to_csv(save_path, index=False, encoding="utf-8-sig")

    return  df_deduplicated




data_path = 'C:/Users/User/Desktop/bigp/AI-DATA/Data_Analysis/Data_ver2/Data/unfair_article_6.csv'
save_path =  "C:/Users/User/Desktop/bigp/AI-DATA/Data_Analysis/Data_ver2/Data/unfair_article_6_first_preprocessing.csv"

print(preprocessing(data_path,save_path))