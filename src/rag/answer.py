# src/rag/answer.py  (예시)
import os, openai

# ① 환경 변수에서 키 읽기 ─ ~/.bashrc 또는 ~/.zshrc 에 export 해두세요
#    export OPENAI_API_KEY="sk-...."
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-3.5-turbo-0125"   # ← 최신 3.5 turbo

def generate_answer_with_feedback(prompt: str) -> str:
    # 1차 초안
    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7    # 필요에 따라 조절
    )
    draft = resp.choices[0].message.content.strip()

    # 2차 자체 리뷰(간단 버전)
    feedback_prompt = f"""
You are an expert reviewer. Please improve the answer below.

QUESTION:
{prompt.split('Question:')[-1].split('Answer:')[0].strip()}

ANSWER DRAFT:
{draft}

Write an improved answer:
"""
    improved = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": feedback_prompt}],
        temperature=0.3
    ).choices[0].message.content.strip()

    return improved