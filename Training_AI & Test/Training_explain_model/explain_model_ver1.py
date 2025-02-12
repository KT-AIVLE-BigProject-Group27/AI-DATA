import openai
import json
import os

# OpenAI API Key 설정
with open('./key/openAI_key.txt', 'r') as file:
    openai.api_key = file.readline().strip()
os.environ['OPENAI_API_KEY'] = openai.api_key
client = openai.OpenAI()

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def explanation_AI(sentence, unfair_label, toxic_label, law=None):
    with open('./Module/Key/openAI_key.txt', 'r') as file:
        openai.api_key = file.readline().strip()
    os.environ['OPENAI_API_KEY'] = openai.api_key
    client = openai.OpenAI()
    if unfair_label == 0 and toxic_label == 0:
        return None

    if unfair_label == 1:
        prompt = f"""
            아래 계약 조항이 **대규모유통업법**을 위반하는지 분석하고, 해당 법 조항(제n조 제m항 제z호)에 따른 위반 여부를 명확하게 설명하세요.

            - 계약 조항: "{sentence}"
            - 관련 법 조항: {law if law else "관련 법 조항 없음"}

            **설명 형식:**  
            - 위반 여부를 명확히 판단하고, 관련 법 조항(제n조 제m항 제z호)을 정확하게 기재하세요.  
            - 제공된 법 조항이 적절하지 않다면, GPT가 판단하여 대규모유통업법의 적절한 조항을 직접 사용하세요.  
            - 위반 사유를 계약 당사자의 불이익과 법적 근거를 들어 논리적으로 설명하세요.  
            - 반드시 문장의 끝을 "~한 이유로 위법 조항입니다."로 마무리하세요.
            - "독소조항"이라는 표현을 사용하지 마세요.

            **예시:**  
            "해당 조항은 대규모유통업법 제n조 제m항 제z호를 위반하였습니다. 이유는 ~~~ 때문입니다. "
        """
    elif toxic_label == 1:
        prompt = f"""
            아래 계약 조항이 독소 조항인지 분석하고, 을(계약 상대방)에게 불리한 조항이라면 그 이유를 설명하세요.

            - 계약 조항: "{sentence}"

            **설명 형식:**  
            - 법 위반 여부가 아닌 을에게 불리한 점을 중심으로 설명하세요.  
            - 을이 불공정한 부담을 지거나 계약상 권리가 제한되는 부분을 강조하세요.  

            **예시:**  
            "이 조항은 ~~한 이유로 을에게 불리한 독소 조항입니다."
        """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": "당신은 계약 조항의 법률 위반 여부와 독소 조항 여부를 분석하는 전문가입니다. "
                        "불공정 조항의 경우, 관련 법 조항을 명확히 기재한 후, 위반 사유를 논리적으로 설명하세요. "
                        "독소 조항의 경우, 을(계약 상대방)이 불리한 점을 중심으로 설명하세요. "
                        "반드시 200 tokens 이하로 작성하세요."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    ).choices[0].message.content

    return response.strip()

def process_data(input_file, output_file):
    data = load_json(input_file)
    for item in data:
        print(item)
        item["explain"] = get_explanation(
            item["Sentence"], item["Unfair"], item["Toxic"], item["law"]
        )
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"✅ 결과가 {output_file} 파일에 저장되었습니다.")


# 실행 (입력 JSON → 설명 생성 → 출력 JSON 저장)
input_json_file = "./Data_Analysis/Data/toxic_or_unfair_identification.json"  # 입력 JSON 파일명
output_json_file = "./Data_Analysis/Data/identification_explain.json"  # 결과 저장 파일명
process_data(input_json_file, output_json_file)
