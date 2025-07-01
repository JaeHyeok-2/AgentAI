# rag/feedback.py
import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")
FEEDBACK_MODEL = "gpt-3.5-turbo-0125"

def apply_feedback(draft: str, original_prompt: str) -> str:
    fb_prompt = f"""
You are an expert reviewer. Please improve the following draft.

Original Question:
{original_prompt.split('Question:')[-1].split('Answer:')[0].strip()}

Draft Answer:
{draft}

Improved Answer:
"""
    resp = openai.ChatCompletion.create(
        model=FEEDBACK_MODEL,
        messages=[{"role": "user", "content": fb_prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()