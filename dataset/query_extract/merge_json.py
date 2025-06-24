# import json

# # 두 파일을 불러오기
# with open('../seperate_data/New_AI_model_no_query_CNAPS_159.json', 'r', encoding='utf-8') as f:
#     data1 = json.load(f)

# with open('../seperate_data/New_AI_model_no_query.json', 'r', encoding='utf-8') as f:
#     data2 = json.load(f)

# standard_cols = [
#     "Model", 
#     "Model Unique Name", 
#     "Category", 
#     "Detailed Category", 
#     "Dataset", 
#     "Paper", 
#     "GitHub", 
#     "HuggingFace", 
#     "Summary"
# ]

# # 키를 기준에 맞게 변환해주는 mapping dictionary 생성
# key_map = {
#     "Github": "GitHub",    # 대소문자 맞추기
#     "Huggingface": "HuggingFace"
# }

# # 모든 데이터 병합
# merged_data = []

# for entry in data1 + data2:
#     normalized_entry = {col: None for col in standard_cols}
#     for key, value in entry.items():
#         standardized_key = key_map.get(key, key)  # 키가 매핑에 없으면 원본 사용
#         if standardized_key in normalized_entry:
#             normalized_entry[standardized_key] = value
#     merged_data.append(normalized_entry)

# # 병합된 데이터를 JSON 파일로 저장
# with open('merged_standard.json', 'w', encoding='utf-8') as f:
#     json.dump(merged_data, f, ensure_ascii=False, indent=2)

# print("✅ merged_standard.json 파일이 성공적으로 저장되었습니다.")



import json

# 파일 경로 설정
queries_path = '/home/cvlab/Desktop/AgentAI/dataset/seperate_data/model_queries_only.json'
user_path = '/home/cvlab/Desktop/AgentAI/dataset/seperate_data/model_queries_CNAPS_159.json'
output_path = 'merged_json.json'

# JSON 파일 읽기
with open(queries_path, 'r', encoding='utf-8') as f:
    queries_json = json.load(f)

with open(user_path, 'r', encoding='utf-8') as f:
    user_json = json.load(f)

# 병합된 데이터를 저장할 리스트
merged_json = []

# queries_json을 빠르게 조회할 수 있도록 인덱스화
queries_index = {entry["Model Unique Name"]: entry for entry in queries_json}

# user_json 기준으로 병합
for entry in user_json:
    model_name = entry["Model Unique Name"]
    merged_entry = entry.copy()  # 기존 필드 유지

    if model_name in queries_index:
        merged_entry["Query1"] = queries_index[model_name].get("Query1")
        merged_entry["Query2"] = queries_index[model_name].get("Query2")
        merged_entry["Query3"] = queries_index[model_name].get("Query3")
    else:
        # queries_json에 없는 경우 기존 쿼리 유지 (entry에 없다면 None이 들어감)
        merged_entry["Query1"] = entry.get("Query1")
        merged_entry["Query2"] = entry.get("Query2")
        merged_entry["Query3"] = entry.get("Query3")

    merged_json.append(merged_entry)

# JSON 파일로 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_json, f, ensure_ascii=False, indent=2)

print("✅ merged_json.json 파일이 성공적으로 저장되었습니다.")