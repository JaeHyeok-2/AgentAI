import pandas as pd
import requests
import feedparser
import time
from tqdm import tqdm
import re

# 세션 재사용 + 헤더 설정
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FetchArxiv/2.0)"})

def extract_arxiv_id_from_url(url):
    """Extract arXiv ID (without version) from PDF URL."""
    match = re.search(r'pdf/(\d{4}\.\d{5})v?\d*', url)
    return match.group(1) if match else None

def fetch_arxiv_summary(arxiv_id, max_attempts=5):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if feed.entries:
                return feed.entries[0].summary.strip().replace('\n', ' ')
            return None
        except requests.exceptions.RequestException as e:
            print(f"[{arxiv_id}] Attempt {attempt}/{max_attempts} failed: {e}")
            time.sleep(2 ** attempt)
    print(f"[{arxiv_id}] ❌ Failed after {max_attempts} attempts.")
    return None

# 📄 데이터 불러오기
df = pd.read_excel("../dataset/arxiv_2023_after.xlsx")

# 🆔 arxiv_id 추출 (v 제거)
df["arxiv_id"] = df["Paper"].apply(extract_arxiv_id_from_url)

# 요약 수집
summaries = []
for arxiv_id in tqdm(df["arxiv_id"]):
    if pd.isna(arxiv_id):
        summaries.append(None)
        continue
    summary = fetch_arxiv_summary(arxiv_id)
    summaries.append(summary)
    time.sleep(3)  # arXiv 권장 요청 간격

# 결과 저장
df["Summary"] = summaries
df.to_excel("arxiv_2023_after_with_summary.xlsx", index=False)