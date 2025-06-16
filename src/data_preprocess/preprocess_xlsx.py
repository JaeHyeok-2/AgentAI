import pandas as pd

# 엑셀 파일 경로
excel_path = "../dataset/New_AI_Model.xlsx"
# 엑셀 파일 불러오기 (모든 시트 불러오기)
excel_data = pd.read_excel(excel_path, sheet_name=None)

# 시트 이름 확인 (예: '2023-2025 New Model'이 주요 시트)
print("시트 목록:", list(excel_data.keys()))

# 주요 시트만 선택
df = excel_data['2023-2025 New Model']

# JSON으로 저장
df.to_json("New_AI_Model.json", orient="records", force_ascii=False, indent=2)