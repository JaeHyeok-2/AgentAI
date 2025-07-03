# src/rag/answer.py
import os
import re
import openai
import anthropic
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL_KEYS = {
    # "chatgpt": "openai/gpt-3.5-turbo-0125",
    "claude": "anthropic/claude-3-5-sonnet-20240620",
    "gemini": "gemini/gemini-2.5-flash"
}

def call_llm(model_key: str, prompt: str, temperature: float = 0.7) -> str:
    if model_key.startswith("openai/"):
        resp = openai.ChatCompletion.create(
            model=model_key.split("/",1)[1],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    elif model_key.startswith("anthropic/"):
        resp = anthropic_client.messages.create(
            model=model_key.split("/",1)[1],
            messages=[{"role": "user", "content": prompt}],
            max_tokens_to_sample=1024,
            temperature=temperature
        )
        return resp["content"].strip()
    elif model_key.startswith("gemini/"):
        resp = gemini_client.chat.completions.create(
            model=model_key.split("/",1)[1],
            messages=[
                {"role": "system", "content": "You are an expert AI scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return resp.choices[0].message["content"].strip()
    else:
        raise ValueError(f"지원되지 않는 모델 키: {model_key}")

def grade_answer_with_llm(answer: str) -> float:
    rubric = (
        "다음 답변을 1점에서 5점 사이로 평가하세요. "
        "기준은: 명확성, 정확성, 논문 인용 반영 여부입니다.\n\n"
        f"Answer:\n{answer}\n\n"
        "최종 점수만 숫자로 출력해주세요 (예: 4)"
    )
    resp = call_llm("openai/gpt-3.5-turbo-0125", rubric, temperature=0)
    m = re.search(r"\b([1-5])\b", resp)
    return float(m.group(1)) if m else 0.0

def generate_multi_with_feedback(
    prompt: str,
    recommended_model_name: str,
    use_feedback: bool = False,
    feedback_threshold: float = 3.0
) -> dict:
    """
    prompt: RAG 기반 생성 프롬프트
    recommended_model_name: RAG에서 추천된 모델 이름
    return: {
      alias: {
        "draft": str,
        "score": float,
        "final": str,
        "refeedback": bool
      }, ...
    }
    """
    prompt_with_rec = (
        f"{prompt}\n\n[Recommended model: {recommended_model_name}]"
        if recommended_model_name else prompt
    )
    results = {}

    for alias, key in MODEL_KEYS.items():
        try:
            draft = call_llm(key, prompt_with_rec)
            score = grade_answer_with_llm(draft)
            final = draft
            refeedback = False

            if use_feedback and score >= feedback_threshold:
                fb_prompt = (
                    "You are an expert reviewer. "
                    "아래 답변을 논문 내용 기반으로 명확하고 정확하게 개선해주세요:\n\n"
                    f"{draft}"
                )
                final = call_llm(MODEL_KEYS["chatgpt"], fb_prompt, temperature=0.3)
                refeedback = True

            results[alias] = {
                "draft": draft,
                "score": score,
                "final": final,
                "refeedback": refeedback
            }

        except Exception as e:
            results[alias] = {"error": str(e)}

    return results