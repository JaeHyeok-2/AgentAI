import os
import openai
from dotenv import load_dotenv

# .env에 OPENAI_API_KEY가 들어 있어야 함
load_dotenv()
openai.api_key = "..."

def call_gpt4o(prompt, system_prompt="You are a helpful assistant.", temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # 또는 gpt-4o-2024-05-13 등 버전 명시
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT-4o API Error:", e)
        return None

# 예시 호출
if __name__ == "__main__":
    user_prompt = "Explain the CNAPS-style AI workflow for enhancing blurry night photos."
    result = call_gpt4o(user_prompt)
    print("\n🧠 GPT-4o Response:\n", result)
