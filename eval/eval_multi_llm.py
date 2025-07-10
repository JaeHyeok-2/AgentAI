import os
import json
import re
from dotenv import load_dotenv

import glob 
load_dotenv()

ROOT_DIR = "/Users/jaeyoung/kuaicv/AgentAI/output/prompts_by_model_query"
MODEL_KEYS = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4-20250514",
    "gemini/gemini-2.5-pro"
]
LLM_NAME_MAP = {
    "gpt-4o": "llm_a",
    "claude-sonnet-4-20250514": "llm_b",
    "gemini/gemini-2.5-pro": "llm_c"
}



# rag/llm.py
import os
from openai import OpenAI
import anthropic
import google.generativeai as genai

# ── API 키 설정 ─────────────────────────
openai_client = OpenAI(  # OpenAI 클래스 인스턴스 생성
    api_key=os.getenv("OPENAI_API_KEY")
)


anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

client = OpenAI()

# ── LLM 호출 헬퍼 ───────────────────────
def call_llm(model_key: str, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:

    """
    model_key 예시
      • openai/gpt-4o-mini
      • anthropic/claude-3-sonnet-20240229
      • claude-3-haiku-20240307     (anthropic/ 생략 허용)
      • gemini/gemini-2.5-flash
    """
    # ---------- OpenAI ----------
    if model_key.startswith("openai/"):
        model_name = model_key.split("/", 1)[1]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}], # the prompt that we manually made with our query
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()

    # ---------- Anthropic Claude ----------
    if model_key.startswith("anthropic/"):
        model_name = model_key.split("/", 1)[1]
    elif model_key.startswith("claude"):
        model_name = model_key                 # alias 허용
    else:
        model_name = None

    if model_name:
        # Claude v1 스타일(Completion) ← 구버전 호환
        if model_name.startswith("claude-") and "-2023" in model_name:
            resp = anthropic_client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens_to_sample=4096,
                temperature=temperature
            )
            return resp["completion"].strip()

        # Claude v3 Messages API
        resp = anthropic_client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )
        return resp.content[0].text.strip()

    # ---------- Google Gemini ----------
    if model_key.startswith("gemini/"):
        model_name = model_key.split("/", 1)[1]
        chat = genai.GenerativeModel(model_name).start_chat(history=[])
        resp = chat.send_message(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 4096
            }
        )
        # import pdb; pdb.set_trace()
        return resp.text.strip()

    # ---------- 알 수 없는 키 ----------
    raise ValueError(f"Unknown model key: {model_key}")

def create_evaluation_prompt(llm_a, llm_b, llm_c):
    system_prompt = "You are an expert evaluator scoring LLM responses based on specific criteria."

    user_prompt = f"""
**Evaluation Criteria (10 points each):**
| Criterion | Description |
|----------------------------------------|--------------------------------------------------|
| 1. Clarity & Readability               | Is the explanation clear and well-organized?    |
| 2. Correctness & Completeness          | Are all required sections addressed?            |
| 3. CNAPS-style Workflow Design         | Is there real branching/merging logic?          |
| 4. Use of Provided Models Only         | Are only the listed models used?                |
| 5. Interpretability & Reasoning        | Is model choice well justified?                 |

—
**RESPONSES TO EVALUATE:**

## llm_a
{llm_a}

—
## llm_b
{llm_b}

—
## llm_c
{llm_c}

—
—
**REQUIRED OUTPUT FORMAT:**
You must provide numerical scores (0–10) for each criterion and each response in the evaluation table.  
**Do not omit this table or alter its structure. It is required for automatic evaluation.**

## Overall Winner: llm_[a/b/c]

### Evaluation Table
| Criterion | llm_a | llm_b | llm_c |
|-----------|-------|-------|-------|
| 1. Clarity & Readability       | /10 | /10 | /10 |
| 2. Correctness & Completeness  | /10 | /10 | /10 |
| 3. CNAPS-style Workflow Design | /10 | /10 | /10 |
| 4. Use of Provided Models Only | /10 | /10 | /10 |
| 5. Interpretability & Reasoning| /10 | /10 | /10 |
| **Total Score**                | /50 | /50 | /50 |


### Brief Justification
- **llm_a**: ...
- **llm_b**: ...
- **llm_c**: ...
"""
    return system_prompt, user_prompt.strip()


def parse_evaluation_result(md_text, model_key):
    """LLM 평가 결과에서 승자, 스코어표, 이유를 추출"""
    vote_match = re.search(r"Overall Winner: (llm_[abc])", md_text)
    winner = vote_match.group(1) if vote_match else None

    # 스코어 테이블 파싱
    score_table = {}
    if model_key.startswith("gemini"):
        total_score_match = re.findall(
            r"\|\s*\*\*Total Score\*\*\s*\|\s*\*\*(\d+)/50\*\*\s*\|\s*\*\*(\d+)/50\*\*\s*\|\s*\*\*(\d+)/50\*\*\s*\|",
            md_text
        )
    else:
        total_score_match = re.findall(
        r"\|\s*(\d{1,2})\s*/?\s*50\s*\|\s*(\d{1,2})\s*/?\s*50\s*\|\s*(\d{1,2})\s*/?\s*50\s*\|?",
        md_text
    )


    
    # total_score_match = re.findall(
    #     r"\|\s*Total Score\s*\|\s*(\d{1,2})\s*/?\s*50\s*\|\s*(\d{1,2})\s*/?\s*50\s*\|\s*(\d{1,2})\s*/?\s*50\s*\|?",
    #     md_text,
    #     flags=re.IGNORECASE
    # )

    # print(model_key) #gpt, claude는 이거로 된다.

    if total_score_match:
        a, b, c = map(int, total_score_match[0])
        score_table = {"llm_a": a, "llm_b": b, "llm_c": c}

    # 간단한 설명 추출
    rationales = {}
    rationale_block = re.search(r"### Brief Justification([\s\S]+)", md_text)
    if rationale_block:
        for key in ["llm_a", "llm_b", "llm_c"]:
            match = re.search(rf"- \*\*{key}\*\*: ([\s\S]+?)(\n- \*\*llm_|$)", rationale_block.group(1))
            if match:
                rationales[key] = match.group(1).strip()
    return winner, score_table, rationales


def update_llm_mapping_json(json_path, votes, scores, rationales):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vote_list = list(votes.values())
    majority_vote = max(set(vote_list), key=vote_list.count)

    # 평균 스코어 기준 가장 우수한 답변
    average_scores = {"llm_a": [], "llm_b": [], "llm_c": []}
    for score in scores.values():
        for k in ["llm_a", "llm_b", "llm_c"]:
            average_scores[k].append(score.get(k, 0))
    mean_scores = {k: sum(v)/len(v) for k, v in average_scores.items()}
    best_by_score = max(mean_scores, key=mean_scores.get)

    data.update({
        "votes": votes,
        "majority_vote": majority_vote,
        "rationales": rationales,
        "scores": scores,
        "best_by_score": best_by_score,
    })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated: {json_path}")


def evaluate_and_update(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    llm_a, llm_b, llm_c = j["responses"]["llm_a"], j["responses"]["llm_b"], j["responses"]["llm_c"]
    system_prompt, user_prompt = create_evaluation_prompt(llm_a, llm_b, llm_c) # generate system_prompt and user_prompt

    votes = {}
    scores = {}
    rationales = {}

    json_basename = os.path.splitext(os.path.basename(json_path))[0]

    for model_key in MODEL_KEYS:
        model_name = model_key.split("/")[1]
        print(f"Evaluating: {model_name}")

        md_result = call_llm(
            model_key,
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )

        vote, score_table, rationale_map = parse_evaluation_result(md_result, model_key)
        votes[model_name] = vote
        scores[model_name] = score_table
        rationales[model_name] = rationale_map.get(vote, "")

        # # ✅ JSON별 디렉토리 안에 모델별 평가 저장
        # output_dir = os.path.join(os.path.dirname(json_path), json_basename, model_name)
        # os.makedirs(output_dir, exist_ok=True)

        # eval_path = os.path.join(output_dir, "eval.md")
        # with open(eval_path, "w", encoding="utf-8") as f:
        #     f.write(md_result)

    update_llm_mapping_json(json_path, votes, scores, rationales)


# ROOT_DIR = "/Users/jaeyoung/kuaicv/AgentAI/output/prompts_by_model_query1_bge_highlevel2"

def is_valid_response_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        # 최소 하나라도 응답이 존재해야 평가 의미 있음
        if "responses" not in j:
            return False

        responses = j["responses"]
        valid_keys = [k for k in ["llm_a", "llm_b", "llm_c"] if k in responses and responses[k].strip()]
        return len(valid_keys) >= 2  # 최소 2개 이상 응답 있어야 비교 가능
    except Exception as e:
        print(f"❌ Invalid JSON: {path} — {e}")
        return False



def main():
    pattern = os.path.join(ROOT_DIR, "**", "**", "*.json")
    print(f"Looking for: {pattern}")
    json_paths = sorted(glob.glob(pattern))
    print(f"Found {len(json_paths)} candidate JSONs")

    for path in json_paths:
        if not is_valid_response_json(path): # check if the merged llm responses have more than two llm responses e.g llm_a and llm_B
            print(f"Skipping invalid or incomplete file: {path}")
            continue
        try:
            print(f"Evaluating: {path}")
            evaluate_and_update(path)
        except Exception as e:
            print(f"Failed on {path}: {e}")
        break
if __name__ == "__main__":
    main()
