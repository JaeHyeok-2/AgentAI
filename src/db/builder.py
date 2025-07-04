import os, json, faiss, numpy as np, sys
from pathlib import Path
from tqdm import tqdm

sys.path.append("..")
from utils.file_io import load_json, save_json
from embedder import embed, MODEL_ID         # ⬅ 기존 임베더
from category2task import CATEGORY2TASK      # ⬅ Detailed→Task 매핑 dict

# ─────────────────────────────────────────────
MODEL_NAME = MODEL_ID.split("/")[-1]
SAVE_DIR   = os.path.join("..", "data", "158_model", MODEL_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== ① Task 목록 JSON (사용자가 제공) =================
TASK_JSON_IN = "/home/cvlab/Desktop/AgentAI/dataset/task_CNAPS.json"   # 경로만 바꿔서 사용
TASK_INDEX_F = os.path.join(SAVE_DIR, "task_index.faiss")
TASK_DOCS_F  = os.path.join(SAVE_DIR, "task_docs.json")

# ===== ② 모델 데이터셋 정의 (변경 없음) ===================
DATASETS = [
    {
        "name": "new_models",
        "input": "/home/cvlab/Desktop/AgentAI/dataset/model_no_query_CNAPS_158.json",
        "index_file": os.path.join(SAVE_DIR, "merged_data.faiss"),
        "output_json": os.path.join(SAVE_DIR, "merged_data.json"),
    }
]

# ─────────────────────────────────────────────
def build_task_index():
    tasks = load_json(TASK_JSON_IN)          # 사용자가 만든 tasks.json
    vecs  = []
    for t in tasks:
        sent = t["description"] + " " + " ".join(t.get("aliases", []))
        vecs.append(embed([sent])[0])
    vecs = np.vstack(vecs).astype("float32")

    index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)
    faiss.write_index(index, TASK_INDEX_F)
    save_json(tasks, TASK_DOCS_F)
    print(f"✅ Task FAISS   → {TASK_INDEX_F}")
    print(f"✅ Task JSON    → {TASK_DOCS_F}")

# ─────────────────────────────────────────────
def build_model_index(ds: dict):
    data = load_json(ds["input"])
    vecs, valid_docs = [], []

    for e in tqdm(data, desc=f"Embedding {ds['name']}"):
        title   = e.get("Model Unique Name", "").strip()
        summary = e.get("Summary_update", "").strip()
        if not summary:
            continue

        # Task Tag 주입
        cat  = e.get("Detailed Category", "").strip()
        task = CATEGORY2TASK.get(cat, "unknown")
        e["Task Tags"] = [task]

        vecs.append(embed([f"{title} {summary}"])[0])
        valid_docs.append(e)

    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)

    faiss.write_index(index, ds["index_file"])
    save_json(valid_docs, ds["output_json"])
    print(f"✅ Model FAISS  → {ds['index_file']}")
    print(f"✅ Model JSON   → {ds['output_json']}")

# ─────────────────────────────────────────────
def main():
    build_task_index()                  # ① Task 인덱스
    for ds in DATASETS:                 # ② 모델 인덱스(경로 그대로)
        build_model_index(ds)

if __name__ == "__main__":
    main()