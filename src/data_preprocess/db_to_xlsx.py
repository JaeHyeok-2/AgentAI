import sqlite3
import pandas as pd

# 1. DB 경로
db_path = "../dataset/arxiv_2023_after.db"

# 2. SQLite 연결
conn = sqlite3.connect(db_path)

# 3. 2023년 이후 데이터 가져오기
query = """
SELECT *
FROM papers
WHERE publish_date >= '2023-01-01'
"""
df = pd.read_sql(query, conn)
conn.close()

# 4. 포맷 재구성
df_converted = pd.DataFrame({
    "Model Unique Name": df["title"],
    "Category": df["topic"],
    "Detailed Category": df["subtopic"],
    "Dataset": "",  # 해당 정보가 DB에 없으므로 공백 처리
    "Paper": df["pdf_url"].str.replace("abs", "pdf") + ".pdf",
    "GitHub": df["code_url"].fillna(""),
    "HuggingFace": ""  # 해당 정보도 DB에 없으므로 공백 처리
})

# 5. Excel로 저장
df_converted.to_excel("arxiv_2023_after.xlsx", index=False)