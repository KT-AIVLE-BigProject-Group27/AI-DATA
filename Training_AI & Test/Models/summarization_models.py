# summarization_models.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_kobart_model(model_name="gogamza/kobart-summarization", device="cuda"):
    """
    KoBART 기반 요약 모델과 토크나이저를 반환합니다.

    모델 설명:
      - KoBART는 BART 계열의 한국어 생성 모델로, 계약서 요약과 같은 태스크에 효과적입니다.
      - 모델 크기: 약 140M 파라미터 (Base)
      - 선정 이유: 한국어에 특화되어 있으며, 복잡한 계약서 텍스트를 효과적으로 요약합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer



