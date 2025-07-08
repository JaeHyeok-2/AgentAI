# rag/llm.py
import os
import openai
import anthropic
import google.generativeai as genai

# ── API 키 설정 ─────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ── LLM 호출 헬퍼 ───────────────────────
def call_llm(model_key: str, prompt: str, temperature: float = 0.7) -> str:
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
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
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