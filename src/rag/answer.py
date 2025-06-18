# src/rag/answer.py  (예시)
import os, openai

# ① 환경 변수에서 키 읽기 ─ ~/.bashrc 또는 ~/.zshrc 에 export 해두세요
#    export OPENAI_API_KEY="sk-...."

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-3.5-turbo-0125"   # ← 최신 3.5 turbo

def generate_answer_with_feedback(prompt: str) -> str:
    # 1차 초안
    rag_prompt = f"""
You are an expert AI scientist.

The user has a question. Please answer based **only on the following retrieved academic papers** from arXiv (from 2023 to 2025).

{prompt.strip()}

Remember:
- Use only the provided context.
- Do not make up facts or refer to papers not listed in the context.
- Structure the answer clearly for a non-technical audience if appropriate.

Answer:
"""
    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": rag_prompt}],
        temperature=0.7
    )
    draft = resp.choices[0].message.content.strip()

    # 2차 리뷰 프롬프트
    feedback_prompt = f"""
You are an expert reviewer. Please improve the following draft.

Original Question:
{prompt.split('Question:')[-1].split('Answer:')[0].strip()}

Draft Answer:
{draft}

Improved Answer:
"""
    improved = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": feedback_prompt}],
        temperature=0.3
    ).choices[0].message.content.strip()

    return improved



# test_prompt = """
# You are an AI scientist.

# A user has asked the following question:
# "I uploaded a selfie and it created a 3D dancing avatar of me—how did it do that?"

# Recommend a suitable AI model for this task, and describe:
# 1. What task the user is performing.
# 2. How this model works in CNAPS AI-like workflows (input → model → result).
# 3. List relevant papers and tools they can use.

# Use only the following selected models and papers for reference:
# Title: AnimateLCM
# Summary: AnimateLCM is a diffusion-based video generation model that enables generating personalized video styles using a single image without requiring user-specific video training data. It fine-tunes latent diffusion with LoRA adapters and temporal modules for consistent video output. The model is efficient and can render videos in a few seconds.
# Paper: https://arxiv.org/pdf/2402.00769.pdf

# Answer:
# """

# if __name__ == "__main__":
#     result = generate_answer_with_feedback(test_prompt)
#     print("💡 Final Answer:\n")
#     print(result)