import os
import json

# 루트 디렉토리 (예: prompts_by_model_query1_bge_mermaid)
BASE_DIR = "/home/cvlab/Desktop/AgentAI/output/prompts_by_model_query1_bge_highlevel"
# BASE_DIR='/home/cvlab/Desktop/AgentAI/output/prompts_by_model_query1_bge_highlevel/WeatherRemoval-CLAIO-DeHaze'
# LLM 익명 키
LLM_KEYS = ["llm_a", "llm_b", "llm_c"]

# 실제 모델명 정렬 기준 (고정 순서: 알파벳 또는 수동 정의 가능)
def sort_llm_files(file_list):
    # 원하는 순서 정의
    key_order = ["claude", "chatgpt", "gemini"]
    return sorted(file_list, key=lambda x: next((i for i, k in enumerate(key_order) if k in x), 99))

# 전체 모델별 매핑 저장
model_llm_mapping = {}

# 모델 폴더 순회
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    # Query1.txt 불러오기
    query_file = os.path.join(folder_path, "Query1.txt")
    if not os.path.exists(query_file):
        continue

    with open(query_file, "r", encoding="utf-8") as f:
        query_text = f.read().strip()

    # LLM 응답 파일만 필터링
    txt_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".txt") and not f.startswith("Query")
    ]
    txt_files = sort_llm_files(txt_files)[:3]

    llm_mapping = {}
    anon_mapping = {}

    for i, fname in enumerate(txt_files):
        llm_key = LLM_KEYS[i]
        file_path = os.path.join(folder_path, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            llm_mapping[llm_key] = f.read().strip()
        anon_mapping[llm_key] = fname.replace(".txt", "")

    # 저장 JSON
    output_json = {
        "query_id": "Query1",
        "query_text": query_text,
        "responses": llm_mapping
    }

    with open(os.path.join(folder_path, f"{folder}.json"), "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    model_llm_mapping[folder] = anon_mapping  # 모델당 llm_x → 실제 모델명

# 전체 매핑 JSON 저장
with open(os.path.join(BASE_DIR, "llm_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(model_llm_mapping, f, indent=2, ensure_ascii=False)

print("✅ 모든 JSON 생성 완료. 매핑 파일: llm_mapping.json")