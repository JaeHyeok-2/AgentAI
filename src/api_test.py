#!/usr/bin/env python3
"""
PROMPT_ROOT 안의 모든 Query*.txt 프롬프트를 찾아
지정한 LLM(OpenAI·Claude·Gemini)을 호출한 뒤
같은 폴더에 응답을 저장한다.
"""

import os, glob, time, traceback, sys
import openai, anthropic, google.generativeai as genai

# ── 환경 변수로부터 API 키 로딩 ─────────────────────────────
openai.api_key      = os.getenv("OPENAI_API_KEY")
anthropic_client    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── 사용자 설정 ────────────────────────────────────────────
PROMPT_ROOT = "output/prompts_by_model_query1_bge"      # 프롬프트 최상위 폴더
LLM_KEYS = {                                           # 결과파일명 : model_key
    "claude_sonnet4.txt": "anthropic/claude-sonnet-4-20250514",
    "gemini_pro25.txt":   "gemini/gemini-2.5-pro",
    # "gpt4o.txt":       "openai/gpt-4o-mini",
}
TEMPERATURE = 0.7
PAUSE_SEC   = 1.0                                      # 연속 호출 간 대기

# ── LLM 호출 함수 ─────────────────────────────────────────
def call_llm(model_key: str, prompt: str, *, temperature: float = 0.7) -> str:
    """model_key 규칙에 따라 LLM 호출, 응답 문자열 반환"""

    # --- OpenAI ------------------------------------------------
    if model_key.startswith("openai/"):
        resp = openai.ChatCompletion.create(
            model=model_key.split("/", 1)[1],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    # --- Anthropic (Claude) -----------------------------------
    if model_key.startswith(("anthropic/", "claude")):
        name = model_key.split("/", 1)[1] if "/" in model_key else model_key
        resp = anthropic_client.messages.create(
            model=name,
            max_tokens=2048,
            temperature=temperature,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return resp.content[0].text.strip()

    # --- Google Gemini ---------------------------------------
    if model_key.startswith("gemini/"):
        real_name = model_key.split("/", 1)[1]
        model     = genai.GenerativeModel(real_name)
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 4096,
                },
            )
            if resp and resp.parts:
                return resp.text.strip()
            # 응답이 비었거나 안전 필터에 걸린 경우
            msg = "(빈 응답 또는 Safety Filter)"
            if resp.prompt_feedback and resp.prompt_feedback.safety_ratings:
                msg += f" 안전등급: {resp.prompt_feedback.safety_ratings}"
            print(f"⚠️  Gemini 경고: {msg}")
            return ""
        except Exception as e:
            print(f"❌ Gemini 호출 오류({model_key}): {e}")
            traceback.print_exc()
            return ""

    # --- 지원되지 않는 모델 키 -------------------------------
    raise ValueError(f"[call_llm] 지원되지 않는 model_key: {model_key}")

# ── 프롬프트 하나 처리 ────────────────────────────────────
def run_on_prompt(prompt_path: str) -> None:
    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read()

    folder = os.path.dirname(prompt_path)

    for out_fname, mkey in LLM_KEYS.items():
        out_path = os.path.join(folder, out_fname)
        if os.path.exists(out_path):
            print("⏭️  이미 존재, 건너뜀:", out_path)
            continue
        try:
            print(f"🔸 {mkey} → {out_path}")
            answer = call_llm(mkey, prompt, temperature=TEMPERATURE)
            with open(out_path, "w", encoding="utf-8") as fo:
                fo.write(answer)
            time.sleep(PAUSE_SEC)
        except Exception as e:
            print(f"❌ {mkey} 처리 중 오류:", e)
            traceback.print_exc()

# ── 배치 실행 ─────────────────────────────────────────────
def main(root: str) -> None:
    pattern = os.path.join(root, "**", "Query*.txt")
    prompts = glob.glob(pattern, recursive=True)
    print(f"▶️  {len(prompts)}개 프롬프트 발견")
    for p in prompts:
        run_on_prompt(p)

# ── CLI 엔트리포인트 ─────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CNAPS LLM 배치 실행")
    ap.add_argument("--root", default=PROMPT_ROOT,
                    help="Query*.txt 파일들이 있는 최상위 폴더 경로")
    args = ap.parse_args()

    PROMPT_ROOT =  "/home/cvlab/Desktop/AgentAI/output/prompts_by_model_query1_bge/Colorization-DISCO-c0_2"
    main(PROMPT_ROOT)