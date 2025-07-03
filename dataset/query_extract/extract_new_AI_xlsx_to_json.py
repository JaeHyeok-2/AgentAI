#!/usr/bin/env python3
"""
make_arxiv_json.py
──────────────────
엑셀/CSV/TSV → arxiv_only.json 변환 (새로운 모델 정보 기준)
"""

import pandas as pd, json, argparse
from pathlib import Path

KEEP_COLS = [
    "Model Unique Name",
    "Category",
    "Detailed Category",
    "Dataset",
    "Paper",
    "Github",
    "HuggingFace",
    "Query1",
    # "Query2",
    # "Query3",
    "Summary_update"
]

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    sep = "," if path.suffix == ".csv" else "\t"
    return pd.read_csv(path, sep=sep)

def main(src_file: str):
    src = Path(src_file)
    df = load_table(src)

    # 컬럼 누락 여부 확인
    missing_cols = set(KEEP_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"⚠️ 다음 컬럼이 누락되었습니다: {missing_cols}")

    df = df[KEEP_COLS]

    # 결측값 (NaN) → None으로 변환
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    out_path = src.with_name("AI_models_CNAPS_159_query1.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ arxiv_only.json 저장 완료 → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="입력 파일 (.xlsx/.csv/.tsv)")
    args = parser.parse_args()
    main(args.file)