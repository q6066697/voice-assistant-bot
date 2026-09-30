"""Нормализация текста перед синтезом речи: латинские термины -> русское произношение.

Русский голос Piper не умеет читать латиницу: например, «RAG» звучит как «ай и джи».
Поэтому перед озвучкой заменяем известные термины на то, как их произносят вслух.
"""
import re

PRONUNCIATION = {
    "RAG": "раг",
    "LLM": "эл эл эм",
    "AI": "эй ай",
    "API": "эй пи ай",
    "GitHub": "гитхаб",
    "Python": "пайтон",
    "speech-to-text": "спич ту текст",
    "text-to-speech": "текст ту спич",
    "STT": "эс ти ти",
    "TTS": "ти ти эс",
}


def normalize_for_tts(text: str) -> str:
    # Сначала длинные термины, чтобы «text-to-speech» не разбился на части
    for term in sorted(PRONUNCIATION, key=len, reverse=True):
        # \b — граница слова: «AI» не должен заменяться внутри других слов
        pattern = r"\b" + re.escape(term) + r"\b"
        text = re.sub(pattern, PRONUNCIATION[term], text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    sample = "Объясни коротко, что такое RAG и зачем он нужен? Посмотри GitHub и text-to-speech."
    print(normalize_for_tts(sample))


# Обратная задача: исправление результата распознавания под предметную область.
# Словарь зависит от домена: «рак» -> «RAG» верно для IT-ассистента,
# но было бы ошибкой в медицинском боте.
STT_CORRECTIONS = {
    "рак": "RAG",   # «RAG» произносится [рак] из-за оглушения на конце слова
    "раг": "RAG",
}


def correct_stt(text: str) -> str:
    for wrong, right in STT_CORRECTIONS.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    return text
