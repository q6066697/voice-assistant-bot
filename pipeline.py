"""Голосовой ассистент: голос -> текст (Whisper) -> ответ (LLM) -> голос (Piper).

Этапы: распознавание -> исправление терминов -> проверка аббревиатур ->
LLM (или уточняющий вопрос) -> нормализация произношения -> синтез.
"""
import sys
import time
import wave

from faster_whisper import WhisperModel
from piper import PiperVoice

from guard import clarification
from llm import answer
from normalize import correct_stt, normalize_for_tts

WHISPER_SIZE = "base"  # на коротких вопросах в ~4 раза быстрее small, см. EXPERIMENTS.md
WHISPER_BEAM = 5
WHISPER_PROMPT = "Термины: GitHub, портфолио, speech-to-text, text-to-speech, AI, LLM, RAG."
VOICE_PATH = "voices/ru_RU-irina-medium.onnx"

# Модели загружаются один раз: в боте они будут жить всё время работы
print("Загружаю модели...")
stt_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
tts_voice = PiperVoice.load(VOICE_PATH)


def speech_to_text(path: str) -> str:
    segments, _ = stt_model.transcribe(
        path, language="ru", vad_filter=True,
        beam_size=WHISPER_BEAM, initial_prompt=WHISPER_PROMPT,
    )
    raw = " ".join(s.text.strip() for s in segments)
    return correct_stt(raw)  # «рак» -> «RAG» и т.п.


def text_to_speech(text: str, out_path: str) -> None:
    with wave.open(out_path, "wb") as wav_file:
        tts_voice.synthesize_wav(normalize_for_tts(text), wav_file)


def run(in_path: str, out_path: str = "samples/answer.wav") -> None:
    t0 = time.perf_counter()
    question = speech_to_text(in_path)
    t1 = time.perf_counter()

    # Незнакомая аббревиатура — переспрашиваем сами, модель не вызываем
    reply = clarification(question) or answer(question)
    t2 = time.perf_counter()

    text_to_speech(reply, out_path)
    t3 = time.perf_counter()

    print(f"\nВопрос: {question}")
    print(f"Ответ:  {reply}")
    print(f"\nSTT: {t1 - t0:.1f} с | LLM: {t2 - t1:.1f} с | TTS: {t3 - t2:.1f} с | "
          f"итого: {t3 - t0:.1f} с")
    print(f"Аудио ответа: {out_path}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "samples/question.wav")
