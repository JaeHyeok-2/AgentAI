from typing import Optional
import pandas as pd
import requests, feedparser, time, re
from tqdm import tqdm

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FetchArxiv/2.0)"})

def extract_arxiv_id_from_url(url: str):
    m = re.search(r'pdf/(\d{4}\.\d{5})v?\d*', str(url))
    return m.group(1) if m else None

def fetch_arxiv_summary(arxiv_id: str, max_attempts: int = 5):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if feed.entries:
                return feed.entries[0].summary.strip().replace("\n", " ")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[{arxiv_id}] {attempt}/{max_attempts} failed: {e}")
            time.sleep(2 ** attempt)
    print(f"[{arxiv_id}] ❌ gave up after {max_attempts} attempts")
    return None

def run_scrape(max_rows: Optional[int] = 100):
    """max_rows=None이면 전체 데이터(14 889개)를 모두 처리."""
    df = pd.read_excel("../dataset/arxiv_2023_after.xlsx")
    df["arxiv_id"] = df["Paper"].apply(extract_arxiv_id_from_url)

    # ▶ 샘플링
    if max_rows is not None:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    summaries = []
    for arxiv_id in tqdm(df["arxiv_id"]):
        if pd.isna(arxiv_id):
            summaries.append(None)
            continue
        summaries.append(fetch_arxiv_summary(arxiv_id))
        time.sleep(3)  # arXiv API 권장 대기

    df["Summary"] = summaries
    out_name = (
        f"arxiv_sample_{max_rows}.xlsx"
        if max_rows is not None else "arxiv_2023_after_with_summary.xlsx"
    )
    df.to_excel(out_name, index=False)
    print(f"✅ Saved → {out_name}")

if __name__ == "__main__":
    run_scrape(max_rows=None)   # ← 먼저 100개만 테스트