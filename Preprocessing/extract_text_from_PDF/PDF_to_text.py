# pip install pymupdf

import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path, txt_output_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()

    text = text.replace('\n', ' ')
    text = re.sub(r'-\s?\d+\s?-', '', text)

    with open(txt_output_path, 'w', encoding='utf-8') as file:
        file.write(text)

    print(text)
    return text

pdf_path = r"./Data_Analysis/Contract/example.pdf"  # PDF 파일 경로
txt_output_path = r"./Data_Analysis/Contract/example(pdf_to_txt).txt"  # 출력될 텍스트 파일 경로
text = extract_text_from_pdf(pdf_path, txt_output_path)


