import os
import json

# 루트 디렉토리 (예: prompts_by_model_query1_bge_mermaid)
BASE_DIR = "/Users/jaeyoung/kuaicv/AgentAI/output/prompts_by_model_query"
# BASE_DIR='/home/cvlab/Desktop/AgentAI/output/prompts_by_model_query1_bge_highlevel/WeatherRemoval-CLAIO-DeHaze'
# LLM 익명 키
LLM_KEYS = ["llm_a", "llm_b", "llm_c"]

# 실제 모델명 정렬 기준 (고정 순서: 알파벳 또는 수동 정의 가능)
def sort_llm_files(file_list):
    # 원하는 순서 정의
    key_order = ["claude", "chatgpt", "gemini"]
    return sorted(file_list, key=lambda x: next((i for i, k in enumerate(key_order) if k in x), 99))

query_dirs = []
model_llm_mapping = {}

# 각 모델 디렉토리 탐색
for model_dir in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model_dir)
    if not os.path.isdir(model_path):
        continue
    
    # 그 안에 있는 Query 디렉토리 탐색
    for sub_dir in os.listdir(model_path):
        if sub_dir.lower().startswith("query"):
            full_query_path = os.path.join(model_path, sub_dir)
            if os.path.isdir(full_query_path):
                query_dirs.append(full_query_path)
    

# 모델 폴더 순회
for folder_path in query_dirs:

    query_number = os.path.basename(folder_path) #query_number : Query1 or Query2 or Query3

    query_path = os.path.join(folder_path, f"{query_number}.txt")

    model_dir_path = os.path.dirname(folder_path)

    model_dir_name = os.path.basename(model_dir_path)

    with open(query_path, "r", encoding="utf-8") as f:
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
        "query_id": f"{query_number}",
        "query_text": query_text,
        "responses": llm_mapping
    }

    with open(os.path.join(folder_path, f"{model_dir_name}.json"), "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    model_llm_mapping[model_dir_name] = anon_mapping  # 모델당 llm_x → 실제 모델명

# 전체 매핑 JSON 저장
with open(os.path.join(BASE_DIR, "llm_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(model_llm_mapping, f, indent=2, ensure_ascii=False)

print("모든 JSON 생성 완료. 매핑 파일: llm_mapping.json")