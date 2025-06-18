#!/usr/bin/env python3
"""
make_arxiv_json.py
──────────────────
엑셀/CSV/TSV → arxiv_only.json (간단 스키마) 변환
"""

import pandas as pd, json, argparse
from pathlib import Path

KEEP_COLS = ["Model Unique Name",
             "Category",
             "Detailed Category",
             "Dataset",
             "Paper",
             "GitHub",
             "HuggingFace",
             "arxiv_id",
             "Summary"]

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    sep = "," if path.suffix == ".csv" else "\t"
    return pd.read_csv(path, sep=sep)

def main(src_file: str):
    src = Path(src_file)
    df  = load_table(src)

    # 필요한 열만 선택
    df = df[KEEP_COLS]

    out_path = src.with_name("arxiv_sample_2023_after.json")
    out_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"✅ arxiv_only.json 저장 완료 →  {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="arxiv_2023_after_with_summary.xlsx / .csv / .tsv")
    args = parser.parse_args()
    main(args.file)