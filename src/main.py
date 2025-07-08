#!/usr/bin/env python3
"""
PROMPT_ROOT 아래에 있는 모든 Query*.txt 파일을 순회하며
지정한 여러 LLM을 호출하고, 각 프롬프트와 같은 폴더에
LLM 응답을 파일로 저장합니다.
"""
import os, glob, time, traceback, sys
sys.path.append("src")                # 프로젝트 루트에서 실행할 때 필요한 경로 추가

from rag.llm import call_llm          # 기존에 작성한 LLM 호출 헬퍼 사용

# ── 사용자 설정 영역 ──────────────────────────────────────────
PROMPT_ROOT = "../output/prompts_by_model_query1_bge_mermaid"  # 프롬프트들이 들어 있는 최상위 폴더

# 저장할 파일명 : call_llm 에 넘길 model_key  매핑
LLM_KEYS = {
    "claude_sonnet4.txt": "anthropic/claude-sonnet-4-20250514",
    "gemini_pro.txt": "gemini/gemini-2.5-pro",
    # "gpt4o.txt":        "openai/gpt-4o-mini",   # 필요 시 주석 해제
}

TEMPERATURE = 0.7   # LLM 샘플링 파라미터
PAUSE_SEC   = 1.0   # API 연속 호출 시 잠시 대기(요금·레이트 제한 완화)
# ────────────────────────────────────────────────────────────


def run_on_prompt(prompt_path: str) -> None:
    """하나의 Query*.txt 프롬프트 파일에 대해 여러 LLM을 호출한다."""
    # 프롬프트 읽기
    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read()

    folder = os.path.dirname(prompt_path)  # 동일 폴더에 결과 저장

    for out_name, model_key in LLM_KEYS.items():
        out_path = os.path.join(folder, out_name)

        # 이미 결과 파일이 있으면 스킵 (중복 호출 방지)
        if os.path.exists(out_path):
            print("⚠️  이미 존재하여 건너뜀:", out_path)
            continue

        try:
            print(f"🔸 {model_key} → {out_path}")
            answer = call_llm(model_key, prompt, temperature=TEMPERATURE)

            with open(out_path, "w", encoding="utf-8") as fo:
                fo.write(answer)

            time.sleep(PAUSE_SEC)

        except Exception as e:
            print("❌ 오류:", model_key, e)
            traceback.print_exc()


def main() -> None:
    # PROMPT_ROOT 아래 모든 Query*.txt 파일 검색(재귀)
    pattern = os.path.join(PROMPT_ROOT, "**", "Query*.txt")
    prompt_files = glob.glob(pattern, recursive=True)

    print(f"▶️  총 {len(prompt_files)}개의 프롬프트 파일 발견")
    for p in prompt_files:
        run_on_prompt(p)


if __name__ == "__main__":
    main()