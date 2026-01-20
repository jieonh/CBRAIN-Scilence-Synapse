import json
import os
import threading
import urllib.request
import urllib.error

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")
API_KEY_PATH = os.path.join(os.path.dirname(__file__), "openai_api_key.txt")


def load_prompt_template():
    with open(PROMPT_PATH, "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def load_openai_api_key():
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    if os.path.exists(API_KEY_PATH):
        with open(API_KEY_PATH, "r", encoding="utf-8") as key_file:
            return key_file.read().strip()
    return ""


def build_llm_prompt(initials_text, category_text):
    template = load_prompt_template()
    return template.format(category_text=category_text, initials_text=initials_text)


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


def send_openai_request(prompt):
    api_key = load_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set (env or openai_api_key.txt)")
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(OPENAI_URL, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
    result = json.loads(raw)
    choices = result.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return (message.get("content") or "").strip()


def llm_worker(llm_queue, state):
    while True:
        item = llm_queue.get()
        if item is None:
            break
        prompt = build_llm_prompt(item["initials"], item["category"])
        try:
            if LLM_PROVIDER == "ollama":
                answer = send_ollama_request(prompt)
            elif LLM_PROVIDER == "openai":
                answer = send_openai_request(prompt)
            else:
                answer = "LLM_PROVIDER not supported"
            state["last_response"] = answer or "(no response)"
            state["last_error"] = ""
        except urllib.error.HTTPError as exc:
            error_url = OPENAI_URL if LLM_PROVIDER == "openai" else OLLAMA_URL
            state["last_error"] = f"LLM HTTP {exc.code} ({exc.reason}) - {error_url}"
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
