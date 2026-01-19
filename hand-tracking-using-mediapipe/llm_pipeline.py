import json
import os
import threading
import urllib.request
import urllib.error

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def build_llm_prompt(initials_text, category_text):
    return (
        "다음은 스피드퀴즈의 정답 초성 및 카테고리입니다.\n"
        "초성과 카테고리를 기반으로 가장 그럴듯한 정답 단어를 추론해 주세요.\n"
        f"- 카테고리: {category_text}\n"
        f"- 초성: {initials_text}\n"
        "확률이 높은 순서로 TOP 5를 제시하고, 각 단어의 확률(%)을 함께 써 주세요.\n"
        "형식: 1) 단어 - 40% 처럼 한 줄씩."
    )


def send_ollama_request(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    result = json.loads(raw)
    return result.get("response", "").strip()


def llm_worker(llm_queue, state):
    while True:
        item = llm_queue.get()
        if item is None:
            break
        prompt = build_llm_prompt(item["initials"], item["category"])
        try:
            if LLM_PROVIDER == "ollama":
                answer = send_ollama_request(prompt)
            else:
                answer = "LLM_PROVIDER not supported"
            state["last_response"] = answer or "(no response)"
            state["last_error"] = ""
        except urllib.error.HTTPError as exc:
            state["last_error"] = f"LLM HTTP {exc.code} ({exc.reason}) - {OLLAMA_URL}"
        except Exception as exc:
            state["last_error"] = f"LLM error: {exc}"
        finally:
            llm_queue.task_done()


def start_llm_worker(llm_queue, state):
    worker_thread = threading.Thread(
        target=llm_worker, args=(llm_queue, state), daemon=True
    )
    worker_thread.start()
    return worker_thread
