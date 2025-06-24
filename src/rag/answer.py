import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-3.5-turbo-0125"

def generate_answer_with_feedback(prompt: str, model_name: str) -> str:
    rag_prompt = f"""
You are an expert AI scientist.

The user has a question. Please answer based **only on the following retrieved academic papers** from arXiv (from 2023 to 2025).

The recommended AI model for this task is **{model_name}**.

{prompt.strip()}

Remember:
- Start your answer by explicitly recommending the model: {model_name}.
- Structure the answer into numbered steps for clarity.
- Use only the provided context.
- Do not make up facts or refer to papers not listed.
- Make your explanation clear and suitable for non-expert users.

Answer:
"""

    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": rag_prompt}],
        temperature=0.7
    )
    draft = resp.choices[0].message.content.strip()

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