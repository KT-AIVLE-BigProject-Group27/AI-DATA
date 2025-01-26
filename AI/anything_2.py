import re
import pandas as pd
import os

def preprocess_and_split(text):
    text = text.replace('*', '').replace('◆', '').replace('◇', '')
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    text = text.replace('\n', ' ').replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = re.sub(r'\([^가-힣]*[\u4E00-\u9FFF\u3040-\u30FF]+[^가-힣]*\)', '', text)  # 괄호 안 한자/일본어 제거
    text = re.sub(r'[\u4E00-\u9FFF\u3040-\u30FF]+', '', text)  # 한자/일본어 제거

    chunks = [text[i:i + 10000] for i in range(0, len(text), 10000)]
    result = []

    for chunk in chunks:
        buffer = ""
        is_open_double_quote = False
        is_open_single_quote = False

        for char in chunk:
            buffer += char

            # 따옴표 상태 관리
            if char == '"':
                is_open_double_quote = not is_open_double_quote
            elif char == "'":
                is_open_single_quote = not is_open_single_quote

            # 분리 조건 확인
            if len(buffer) >= 250:
                if char in ['.', '!', '?'] and not is_open_double_quote and not is_open_single_quote:
                    result.append(buffer.strip())
                    buffer = ""
                elif char in ['"'] and  buffer[-3] in ['"']:
                    result.append(buffer[:-1].strip())
                    buffer = '"'
                elif char in ["'"] and  buffer[-3] in ["'"]:
                    result.append(buffer[:-1].strip())
                    buffer = "'"
                elif re.search(r'\.\.\.|!!|\?\?', buffer):  # 반복된 기호 처리
                    match = re.search(r'(\.\.\.|!!|\?\?)', buffer)
                    split_index = match.end()
                    result.append(buffer[:split_index].strip())
                    buffer = buffer[split_index:].strip()
                # else:
                #     print(f'[{len(buffer)}]: d-{is_open_double_quote} s-{is_open_single_quote} / {buffer}')
    return result

folder_path = 'C:/Users/user/Desktop/데이터_1만자/'
result = []
books = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # .txt 파일만 선택
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            split_text = preprocess_and_split(text)
            result.extend(split_text)
            books.extend([filename]*len(split_text))

df_split_sentences = pd.DataFrame({'book': books, 'Sentence': result})
df_split_sentences.to_csv("C:/Users/user/Desktop/split_sentences.csv", index=False, encoding='utf-8-sig')




##################################################################
folder_path = 'C:/Users/user/Desktop/데이터_1만자/'
result = []
with open(folder_path + '남주와 여주가 헤어졌다 3권 (데이데이) (Z-Library).txt', 'r', encoding='utf-8') as file:
    text = file.read()  # 파일 내용 읽기
    result.extend(preprocess_and_split(text))  # 바로 처리하여 결과에 추가

for resul in result:
    print(f'[{len(resul)}]:{resul}')