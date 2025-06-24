#!/usr/bin/env python3
"""
split_queries.py
----------------
New_AI_Model.json을 읽어
  1) 모델-별 Query1/2/3만 추출한 model_queries_only.json
  2) Query 필드를 뺀 models_no_queries.json
두 파일로 저장한다.
"""

import json
import argparse
from pathlib import Path

def split_json(src: Path):
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries_only, models_no_queries = [], []

    for d in data:
        # ① Query 전용 레코드
        queries_only.append({
            "Model Unique Name": d.get("Model Unique Name"),
            "Query1": d.get("Query1"),
            "Query2": d.get("Query2"),
            "Query3": d.get("Query3")
        })
        # ② Query 제거본
        cleaned = {k: v for k, v in d.items() if not k.startswith("Query")}
        models_no_queries.append(cleaned)

    out_queries = src.with_name("model_queries_CNAPS_159.json")
    out_models  = src.with_name("New_AI_model_no_query_CNAPS_159.json")

    out_queries.write_text(json.dumps(queries_only, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    out_models.write_text(json.dumps(models_no_queries, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"✅ saved → {out_queries}")
    print(f"✅ saved → {out_models}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="path to New_AI_Model.json")
    args = parser.parse_args()
    split_json(Path(args.json_path).expanduser())