import os
import json

def find_file_case_insensitive(directory, target_filename_lower):
    """
    폴더 내에서 대소문자 구분 없이 파일명을 찾아 반환
    """
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if filename.lower() == target_filename_lower:
            return os.path.join(directory, filename)
    return None

def read_answer_file(base_path, subfolder, filename):
    path = find_file_case_insensitive(os.path.join(base_path, subfolder), filename.lower())
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def process_model_folder(model_folder_path):
    model_name = os.path.basename(model_folder_path)
    model_json = {"model": model_name}
    for qnum in range(1, 4):
        qkey = f"query{qnum}"
        model_json[qkey] = {
            "llm_A": read_answer_file(model_folder_path, "gpt_answer", f"query{qnum}_answer.txt"),
            "llm_B": read_answer_file(model_folder_path, "gemini_answer", f"query{qnum}_answer.txt"),
            "llm_C": read_answer_file(model_folder_path, "claude_answer", f"query{qnum}_answer.txt"),
        }
    return model_json

def process_all_models(root_dir):
    result = []
    for folder in os.listdir(root_dir):
        model_path = os.path.join(root_dir, folder)
        if os.path.isdir(model_path):
            result.append(process_model_folder(model_path))
    return result

# 사용 예
base_dir = "./"  # 모델 폴더들이 있는 상위 폴더 경로로 수정
all_model_data = process_all_models(base_dir)

# JSON으로 저장
with open("all_model_answers.json", "w", encoding="utf-8") as f:
    json.dump(all_model_data, f, indent=2, ensure_ascii=False)
