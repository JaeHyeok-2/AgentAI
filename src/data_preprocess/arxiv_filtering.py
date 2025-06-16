### 2023년도 이후의 논문만을 위해서 이전의 것들은 모두 필터링 할 예정 

import sqlite3
import pandas as pd

# 원본 DB 경로
db_path = "../dataset/arxiv_2020_2025.db"

# DB 연결
conn = sqlite3.connect(db_path)

# 2023년 이후 데이터 쿼리
query = """
SELECT *
FROM papers
WHERE publish_date >= '2023-01-01'
"""

# 쿼리 실행
df = pd.read_sql(query, conn)

# 원본 DB 연결 종료
conn.close()

# 새로운 DB에 저장
filtered_db_path = "arxiv_2023_after.db"
conn_out = sqlite3.connect(filtered_db_path)
df.to_sql("papers", conn_out, if_exists="replace", index=False)
conn_out.close()