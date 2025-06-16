import openai

openai.api_key = "your-openai-api-key"

def generate_answer_with_feedback(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    draft = response["choices"][0]["message"]["content"]

    feedback_prompt = f"""
You are an AI reviewer. Review and improve the following answer.

Question:
{prompt.split('Question: ')[-1].split('Answer:')[0].strip()}

Original Answer:
{draft}

Improved Answer:
"""
    final = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": feedback_prompt}]
    )["choices"][0]["message"]["content"]

    return final