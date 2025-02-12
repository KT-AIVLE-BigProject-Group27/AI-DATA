# pip install pywin32
import os
import win32com.client

# 원본 폴더 (HWP 파일 위치)
source_dir = r"C:\Users\user\Desktop\AI-DATA\Data_Analysis\Contract"
# 변환된 파일을 저장할 폴더
target_dir = os.path.join(source_dir, "convert_test")

# 변환 폴더가 없으면 생성
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 한컴오피스 자동화 객체 생성
hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")  # 보안 경고 방지

# 원본 폴더에서 모든 .hwp 파일 검색
for file_name in os.listdir(source_dir):
    if file_name.endswith(".hwp"):  # 확장자 필터링
        hwp_path = os.path.join(source_dir, file_name)  # 원본 파일 경로
        pdf_path = os.path.join(target_dir, file_name.replace(".hwp", ".pdf"))  # 저장 경로

        try:
            # HWP 파일 열기
            hwp.Open(hwp_path)

            # PDF 변환 설정 (SaveAs 활용)
            hwp.SaveAs(pdf_path, "PDF")

            print(f"변환 완료: {pdf_path}")
        except Exception as e:
            print(f"변환 실패: {hwp_path} - 오류: {e}")

# 한컴오피스 종료
hwp.Quit()


