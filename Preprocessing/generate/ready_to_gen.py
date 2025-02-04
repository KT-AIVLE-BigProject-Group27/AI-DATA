import re
import json
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

def contract_to_articles(text):
    # "제n조" 단위로 텍스트를 분리
    pattern = r'(제\d+조(?!\S))'  # "제n조" 뒤에 공백이 있거나 끝났을 때
    matches = re.split(pattern, text)

    data = {}
    section_counter = {}  # 각 "제n조"의 중복 횟수를 추적하기 위한 딕셔너리
    for i in range(1, len(matches), 2):
        section_title = matches[i].strip()
        section_content = matches[i + 1].strip()

        # "제n조" 번호 추출
        section_num = re.match(r'제(\d+)조', section_title).groups()[0]

        # 중복 처리
        if section_num in data:
            if section_num in section_counter:
                section_counter[section_num] += 1
            else:
                section_counter[section_num] = 2
            new_title = f"{section_num}_{section_counter[section_num]}"
        else:
            section_counter[section_num] = 1
            new_title = section_num

        data[new_title] = section_content

    def split_sentences(text):
        # 문장을 분리만 수행
        return re.split(r'(\n\n)', text)

    # "제n조"와 "제n조의m"을 그룹화하여 처리하는 함수
    def group_content_sections(data):
        grouped_data = {}

        temp_content = {}  # 세부 항목들 임시 저장

        for key, value in data.items():
            content_sentences = split_sentences(value.strip())  # 문장 분리 수행

            clean_value = re.sub(r'\n\n', '', value.strip())
            clean_value = re.sub(r'\"갑\"', '갑', clean_value)  # \"갑\"을 갑으로 변환
            clean_value = re.sub(r'\"을\"', '을', clean_value)  # \"을\"을 을로 변환
            clean_value = re.sub(r'\\"([^"]+)\\"', r"'\1'", clean_value)
            clean_value = re.sub(r'\"([^"]+)\"', r"'\1'", clean_value)

            # "제n조" 부분을 n으로만 추출하여 저장
            grouped_data[key] = [f"제{key}조 {clean_value}"]

            # "제n조의m" 형식 처리
            temp_key = None
            for sentence in content_sentences:
                sentence = re.sub(r'\n\n', '', sentence.strip())
                sentence = re.sub(r'\"갑\"', '갑', sentence)
                sentence = re.sub(r'\"을\"', '을', sentence)
                sentence = re.sub(r'\\"([^"]+)\\"', r"'\1'", sentence)
                sentence = re.sub(r'\"([^"]+)\"', r"'\1'", sentence)
                match_sub_section = re.match(r'제(\d+)조의(\d+)', sentence)  # "제n조의m" 찾기
                if match_sub_section:
                    # 세부 항목 처리
                    num, sub_num = match_sub_section.groups()
                    temp_key = f"{num}-{sub_num}"
                    if temp_key not in temp_content:
                        temp_content[temp_key] = []
                    temp_content[temp_key].append(sentence.strip())
                    # 추가된 것
                    grouped_data[num] = [s.split(sentence.strip())[0] if sentence.strip() in s else s for s in grouped_data[num]]

                else:
                    match_section = re.match(r'제(\d+)조', sentence)  # "제n조" 구분
                    if match_section:
                        num = match_section.groups()[0]
                        temp_key = f"{num}"
                        if temp_key not in temp_content:
                            temp_content[temp_key] = []
                    if temp_key is not None:
                        temp_content[temp_key].append(sentence.strip())

        # 세부 항목들을 각 조문 바로 뒤에 올 수 있도록 조정
        for key, value in temp_content.items():
            if key in grouped_data:
                grouped_data[key].extend(value)
            else:
                grouped_data[key] = value

        return grouped_data

    def sort_grouped_data(grouped_data):
        # 조항 번호에 따라 정렬
        sorted_grouped_data = {}
        # 정렬 기준: 숫자와 텍스트를 모두 고려하여 정렬
        for key in sorted(grouped_data.keys(), key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)]):
            sorted_grouped_data[key] = grouped_data[key]
        return sorted_grouped_data

    def del_empty_content(output_json):
        for key, value in output_json.items():
            if isinstance(value, list):
                output_json[key] = [item for item in value if item]
        return output_json

    def merge_sentences(grouped_data):
        for key, value in grouped_data.items():
            grouped_data[key] = ' '.join(value)  # 리스트 내부 문장을 하나로 합침
        return grouped_data

    grouped_data = group_content_sections(data)
    grouped_data = sort_grouped_data(grouped_data)
    grouped_data = del_empty_content(grouped_data)
    grouped_data = merge_sentences(grouped_data)

    return grouped_data


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
  # Extract the article title from the content if it's not passed
    if not article_title:
        # Look for the title enclosed in square brackets after the article number
        title_match = re.search(r"제\d+조\s*\[([^\]]+)\]", article_content)
        if title_match:
            article_title = title_match.group(1)  # Extract title (e.g., '목적')
        else:
            article_title = ''  # Default if no title found
    symtostr = {
        "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
        "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10"
    }
    sentences = []
    if '①' in article_content:
        clause_sections = split_once_by_clauses(article_content)
        for i in range(0, len(clause_sections), 2):
            clause_number = clause_sections[i]
            clause_content = clause_sections[i + 1]
            if '1.' in clause_content:
                sub_clause_sections = split_once_by_sub_clauses(clause_content)
                for j in range(0, len(sub_clause_sections), 2):
                    sub_clause_number = sub_clause_sections[j]
                    sub_clause_content = sub_clause_sections[j + 1]
                    sentences.append([article_number.strip(), article_title.strip(), '', symtostr[clause_number].strip(), clause_content.split('1.')[0].strip(), sub_clause_number[0].strip(), sub_clause_content.strip()])
            else:
                sentences.append([article_number.strip(), article_title.strip(), '', symtostr[clause_number].strip(), clause_content.split('①')[0].strip(), '', ''])
    elif '1.' in article_content:
        sub_clause_sections = split_once_by_sub_clauses(article_content)
        for j in range(0, len(sub_clause_sections), 2):
            sub_clause_number = sub_clause_sections[j]
            sub_clause_content = sub_clause_sections[j + 1]
            sentences.append([article_number.strip(), article_title.strip(), article_content.split('1.')[0].strip(), '', '', sub_clause_number.strip(),sub_clause_content.strip()])
    else:
        sentences.append([article_number.strip(),article_title.strip(),article_content.strip(),'','','',''])
    return sentences


file_path = 'Preprocessing/generate/24년 개정 직매입 표준거래계약서(면세점).txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text_file = file.read()

articles = contract_to_articles(text_file)

class SeparateResult(BaseModel):
    article_number: str
    article_title : str
    article_content: str
    clause_number: Optional[str] = None
    clause_content : Optional[str] = None
    sub_clause_number : Optional[str] = None
    sub_clause_content: Optional[str] = None
    
result_sentences = []

for key, value in articles.items():
    # article_to_sentences 결과를 받아오기
        sentences = article_to_sentences(key, '', value)

    # 결과 문장들을 SeparateResult로 저장
        for sentence in sentences:
            result_sentences.append(
                SeparateResult(
                    article_number=sentence[0],
                    article_title=sentence[1],
                    article_content=sentence[2],
                    clause_number = sentence[3],
                    clause_content = sentence[4],
                    sub_clause_number = sentence[5],
                    sub_clause_content = sentence[6]
            )
        )

sp_data =  [result.model_dump() for result in result_sentences]