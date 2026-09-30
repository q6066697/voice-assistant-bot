"""Ответ языковой модели через OpenAI-совместимый API (по умолчанию локальный Ollama)."""
import os
import sys
import time

from openai import OpenAI

# Меняя переменные окружения, можно подключить любой OpenAI-совместимый сервис
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama ключ не проверяет, но поле обязательно
MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "60"))

SYSTEM_PROMPT = (
    "Ты голосовой ассистент по темам AI и программирования. "
    "Отвечай только по-русски, кратко: одно-два предложения. "
    "Не используй markdown, списки, звёздочки и эмодзи — твой ответ будет озвучен. "
    "Числа, даты и сокращения пиши словами, так, как их произносят вслух."
)

# Пример диалога (few-shot): вопрос с английским термином -> короткий ответ по-русски.
# Инструкции «отвечай по-русски» маленькая модель игнорировала, а пример работает сильнее.
FEW_SHOT = [
    {"role": "user", "content": "Что такое API и зачем он нужен?"},
    {"role": "assistant", "content": "Эй пи ай — это способ, которым одни программы "
                                     "обращаются к другим. Он нужен, чтобы сервисы могли "
                                     "обмениваться данными."},
]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def answer(question: str) -> str:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT,
            {"role": "user", "content": f"{question}\n\nОтветь по-русски."},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.3,  # ниже случайность — стабильнее и короче ответы
    )
    elapsed = time.perf_counter() - start
    text = response.choices[0].message.content.strip()
    tokens = response.usage.completion_tokens if response.usage else "?"
    print(f"Модель: {MODEL} | токенов в ответе: {tokens} | время: {elapsed:.1f} с")
    return text


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Что такое распознавание речи?"
    print(answer(q))
