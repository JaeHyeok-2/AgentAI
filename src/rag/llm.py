# rag/llm.py
import os, openai, anthropic
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def call_llm(model_key: str, prompt: str, temperature=0.7) -> str:
    if model_key.startswith("openai/"):
        resp = openai.ChatCompletion.create(
            model=model_key.split("/",1)[1],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    elif model_key.startswith("anthropic/"):
        resp = anthropic_client.completions.create(
            model=model_key.split("/",1)[1],
            prompt=prompt,
            max_tokens_to_sample=1024,
            temperature=temperature
        )
        return resp["completion"].strip()
    elif model_key.startswith("anthropic/claude") or model_key.startswith("claude"):
        # if using simple model name
        resp = anthropic_client.completions.create(
            model=model_key,
            prompt=prompt,
            max_tokens_to_sample=1024,
            temperature=temperature
        )
        return resp["completion"].strip()
    elif model_key.startswith("gemini/"):
        resp = gemini_client.chat.completions.create(
            model=model_key.split("/",1)[1],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return resp.choices[0].message["content"].strip()
    else:
        raise ValueError(f"Unknown model key: {model_key}")