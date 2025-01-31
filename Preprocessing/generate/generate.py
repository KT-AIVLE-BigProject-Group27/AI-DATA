import os
import requests
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
import re
import random
import ready_to_gen


# Google Drive에 저장된 API 키 파일 경로
api_key_path = 'C:/Users/User/Desktop/Keys/openAI_key.txt'

# 파일에서 API 키 읽기
with open(api_key_path, 'r') as file:
    api_key = file.read().strip()

# 환경 변수로 설정
os.environ['OPENAI_API_KEY'] = api_key
client = OpenAI()

import json

# 문장 생성 함수
def generate_contract_clause(sentence: str, sub_clause_content: str = None):
  
  # 랜덤 제품 선택
  global selected_product
  if selected_product is None:
        import random
        products = ["식재료", "음료", "전자제품", "담배", "술", "가방",
          "지갑", "향수","의류","화장품","시계","쥬얼리","샴푸","바디워시"]
        selected_product = random.choice(products)

    # 메시지 준비
  messages = [
        {"role": "system", "content": "당신은 거래 계약서를 작성하는 꼼꼼한 어시스턴트입니다."},
        {"role": "user", "content": f"""
원본 문장은 합법적인 내용하에 약간 수정합니다. 원본 문장의 형식은 반드시 지키세요.
원본 문장안에  ** ~~ ** 을 쓰지 마세요.
원본 문장을 구체적인 예시를 반영하세요. 구체적인 상품, 제품의 시나리오를 작성하세요.
제품의 예시는 다음과 같습니다. 다음 목록 중에 랜덤하게 하나를 고르시면 됩니다.
선택된 제품은 '{selected_product}'입니다. 제품의 예시를 구체적으로 작성하세요. 제품 이름을 적절하게 지어야 합니다.

제품 하나를 지정하고, 맥락을 전체 문장과 통일 시켜 만드세요.
문장 생성은 원본을 참고하여 풍성하게 표현을 바꾸세요. 기존 문장의 의미는 비슷하게 합니다.
풍성하게 표현을 하도록 합니다. 의미는 비슷할 지라도, 다양한 표현을 사용합니다.
원본 문장 내에서 나온 비율을 합법적인 범위 안에서 조금씩 바꾸세요.

다음은 반드시 지켜야하는 규칙입니다.
- 문장 내에 ** ~ **과 같이, 특정 특수문자를 집어넣어서 생성하지말것.
- 문장 내에 '문장', '생성', '생성 결과'와 같은 문구를 쓰지말고, 생성한 결과만 출력할 것.
- 문장 내에 추가 조건을 생성해도 좋으나, 원본의 형식을 무너뜨리지는 말 것.
- 특히, 첫 번째 문장에서, 구체적인 제품을 명시하고, 다음 문장 부터는 첫 번째 문장의 예시와 무조건 동일하게 설정합니다.
- 제품의 예시는 첫번째 줄에만 생성합니다.
- 문장에 **제품**, **상품** 이러한 문구는 절대로 추가하면 안됩니다.
- 모든 빈칸을 모두 채워주세요. 빠짐없이 채워주세요.
- 반드시 생성한 제품명은 한 번 생성했으면 문장 끝까지 모두 동일하게 처리합니다.
- 첫번째에 생성한 제품을 모든 문장이 끝날때까지 가지고 갑니다. 반드시 동일하게 씁니다.
- 첫번째 생성한 제품은 구체적으로 이름을 만들어내도록합니다. 반드시 동일하게 씁니다.
- 반드시 반드시 동일한 제품명만 씁니다.
- 제발 처음 생성한 제품명만 계속 쓰도록 합니다.
- 당신의 판단과 근거, 의견은 배제하도록 합니다.
**문장:**
{sentence}
**제품:**
{selected_product}
**결과:**
"""}
]

    # API 호출
  completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    # 결과 반환
  return completion.choices[0].message.content

def generate_contract_data(article):
    generated_data = []

    # 모든 계약서 항목에 대해 반복
    for contract_data in article:
        clause_content = contract_data['clause_content']  # 원본 clause_content
        sub_clause_content = contract_data['sub_clause_content']  # sub_clause_content
        sub_clause_number = contract_data['sub_clause_number']  # sub_clause_number


        generated_clause_content = generate_contract_clause(sentence=clause_content)

        # 결과 저장
        generated_data.append({
            'article_number': contract_data['article_number'],
            'clause_number': contract_data['clause_number'],
            'sub_clause_number': contract_data['sub_clause_number'],
            'clause_content': generated_clause_content,
            'sub_clause_content': contract_data['sub_clause_content']
        })

    return generated_data

from collections import defaultdict

# 중복된 주 조항을 제거하고, 조항과 서브 조항을 적절하게 정리
def process_contract_data(data):
    result = defaultdict(lambda: {'clause_content': '', 'sub_clauses': []})

    for item in data:
        article_number = item['article_number']
        clause_number = item['clause_number']
        sub_clause_number = item['sub_clause_number']
        clause_content = item['clause_content']
        sub_clause_content = item['sub_clause_content']

        key = f"{article_number}-{clause_number}"

        # 주 조항 처리 (이미 내용이 있으면 중복되지 않도록 처리)
        if result[key]['clause_content'] == '':
            result[key]['clause_content'] = clause_content

        # 서브 조항 처리
        if sub_clause_number:
            sub_clause = f"{int(sub_clause_number)}. {sub_clause_content}"
        else:
            sub_clause = sub_clause_content

        if sub_clause:
            result[key]['sub_clauses'].append(sub_clause)

    # 형식에 맞게 출력
    output = []
    main_clause_counter = 0  # 원형 숫자 카운터 (동적 증가)

    for key, value in result.items():
        article_number, clause_number = key.split('-')
        clause_number = int(clause_number)  # 숫자로 변환하여 동적으로 처리

        # 원형 숫자를 동적으로 사용
        circle_number = chr(9312 + main_clause_counter)  # ①, ②, ③... 를 Unicode로 생성
        output.append(f"{circle_number} {value['clause_content']}")  # 동적 원형 숫자 추가

        # 서브 조항 번호는 원형 숫자 없이 추가
        for sub_clause in value['sub_clauses']:
            output.append(f"  {sub_clause}")

        # 주 조항 번호 증가 (원형 숫자는 동적으로 들어가야 함)
        main_clause_counter += 1
    return "\n".join(output)

## 생성 예시: 컬럼에 해당하는 갯수는 원본 자료 보고 설정할 것(라벨링링)
gen_data_ori = ready_to_gen.sp_data
    
sentences = """
제10조 [서비스 품질유지]
"""
selected_product = None
target = 10

for i in range(50):
    target_data = [item for item in gen_data_ori if item['article_number'] == str(target)]
    target_data = list({frozenset(item.items()): item for item in target_data}.values())

    df_target= pd.DataFrame({
    'article_number': [target] * 3,
    'clause_number': [1, 2, 3],
    'sub_clause_number': [0,0,0],
    'unfair_label': [0] * 3,
    'law_article': [0] * 3,
    'content': [None] * 3
    })

    # 'content' 컬럼에 값을 업데이트
    df_target['content'] = sentences + process_contract_data(generate_contract_data(target_data))

    # 기존 데이터프레임에 새로운 df 추가
    init_data = df_target

init_data.to_csv(f'fair_{target}_article.csv', encoding='utf-8-sig',index=False)

print(init_data)