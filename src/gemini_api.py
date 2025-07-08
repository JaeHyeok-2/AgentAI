import os
import google.generativeai as genai
import traceback

# ── API 키 로딩 ─────────────────────────────────────────────
# 기존 코드와 동일하게 API 키를 환경 변수에서 로드합니다.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Gemini 모델 테스트 함수 ────────────────────────────────
def test_gemini_simple_prompt(model_name: str, test_prompt: str, temperature: float = 0.7) -> None:
    """
    주어진 Gemini 모델에 간단한 프롬프트를 보내고 응답을 출력합니다.
    API 키 및 기본 통신 문제를 확인하는 데 사용됩니다.
    """
    print(f"\n--- {model_name} 모델 테스트 시작 ---")
    print(f"테스트 프롬프트: \"{test_prompt}\"")

    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            test_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 2048, # 넉넉하게 설정
            }
        )

        if resp and resp.parts:
            generated_text = resp.text.strip()
            print("\n✅ 응답 성공:")
            print("--------------------------------------------------")
            print(generated_text)
            print("--------------------------------------------------")
        else:
            print("\n❌ 응답 실패: 모델이 콘텐츠를 생성하지 못했습니다.")
            if resp.prompt_feedback and resp.prompt_feedback.safety_ratings:
                print(f"   안전 필터 정보: {resp.prompt_feedback.safety_ratings}")
            else:
                print("   (안전 필터 정보 없음 또는 다른 이유로 응답 생성 실패)")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc() # 상세한 오류 스택 추적 출력
    finally:
        print(f"--- {model_name} 모델 테스트 종료 ---\n")

# ── 메인 실행 부분 ─────────────────────────────────────────
if __name__ == "__main__":
    # 테스트할 Gemini 모델 이름
    GEMINI_TEST_MODEL = "gemini-2.5-pro" # 또는 "gemini-pro" 등

    # # 간단한 테스트 프롬프트 목록
    # simple_prompts = [
    #     "안녕하세요?",
    #     "간단한 농담 하나 해주세요.",
    #     "서울의 수도는 어디인가요?",
    #     "2+2는 무엇인가요?",
    # ]

    # for prompt in simple_prompts:
    #     test_gemini_simple_prompt(GEMINI_TEST_MODEL, prompt)

    # print("\n--- 복잡한 프롬프트로 테스트 시작 (선택 사항) ---")
    # print("이 부분은 수동으로 원래 프롬프트의 일부를 복사하여 테스트할 수 있습니다.")
    # 원본 프롬프트에서 제약 조건을 제거한 버전을 테스트해 볼 수 있습니다.
    # 예: "You are an expert AI scientist and architect of a multi-module workflow. 사용자 요청: '이 흑백 사진을 색칠하고 싶어. 나무는 자연스러운 녹색 계열로, 사람은 피부색 느낌으로 칠해줘.' 워크플로우를 설계해줘."

    # 예시 (주석 해제 후 사용)
    complex_test_prompt = """
    You are an expert AI scientist and architect of a CNAPS‑style multi‑module workflow.  
Here, CNAPS means a **synapse‑like branching network** of AI models working together—not a simple linear pipeline.

A user asks:
"I want to color this black-and-white photo. Paint the wood naturally with a green color scheme and a person with a skin tone feel."

**Using ONLY the provided models and papers in the context below, do the following:**

1. **Identify the core task or goal** implied by the user’s request.  
2. **Design a CNAPS-style synaptic workflow**:
   - Describe how input is routed to one or more modules.
   - Explain how modules branch, interact, merge, or loop.
   - Define each module’s intermediate and final output formats/include examples.
3. **Justify your design** with references to the papers and tools (include GitHub or ArXiv links listed).


### Recommended AI Models:
- **Colorization-DISCO-c0_2**
  Paper: https://menghanxia.github.io/projects/disco/disco_main.pdf

- **Inpainting-ResShift-Face**
  Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/2ac2eac5098dba08208807b65c5851cc-Paper-Conference.pdf

- **Inpainting-MISF-Places2**
  Paper: https://arxiv.org/pdf/2203.06304

Answer:

    """
    test_gemini_simple_prompt(GEMINI_TEST_MODEL, complex_test_prompt, temperature=0.5)