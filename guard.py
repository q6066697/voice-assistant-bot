"""Проверка вопроса до вызова LLM: незнакомые аббревиатуры -> переспросить.

Локальные модели (qwen2.5 1.5B и 3B) на вопрос с искажённым термином «AIG»
уверенно выдумывали расшифровку. Инструкция «не выдумывай» не помогла, поэтому
проверка сделана в коде: быстро, детерминированно, не зависит от модели.
"""
import re

from normalize import PRONUNCIATION

# Известные термины = словарь произношений + то, что читается без замены
KNOWN_TERMS = {term.upper() for term in PRONUNCIATION} | {"GPU", "CPU", "SQL", "JSON"}

# Аббревиатура: 2–5 заглавных латинских букв подряд
ACRONYM = re.compile(r"\b[A-Z]{2,5}\b")


def unknown_acronyms(question: str) -> list[str]:
    return [a for a in ACRONYM.findall(question) if a not in KNOWN_TERMS]


def clarification(question: str) -> str | None:
    """Вернуть уточняющий вопрос или None, если вопрос можно передать модели."""
    unknown = unknown_acronyms(question)
    if unknown:
        terms = ", ".join(unknown)
        return f"Я не уверен, что правильно расслышал: {terms}. Уточните, пожалуйста, что вы имели в виду?"
    return None


if __name__ == "__main__":
    for q in ["Объясни кратко, что такое AIG и зачем он нужен.",
              "Объясни кратко, что такое RAG и зачем он нужен."]:
        print(q, "->", clarification(q))
