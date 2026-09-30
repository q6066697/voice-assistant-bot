"""Синтез речи из текста через Piper."""
import sys
import time
import wave

from piper import PiperVoice

VOICE_PATH = "voices/ru_RU-irina-medium.onnx"


def synthesize(text: str, out_path: str = "samples/tts_out.wav") -> str:
    voice = PiperVoice.load(VOICE_PATH)

    start = time.perf_counter()
    with wave.open(out_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    elapsed = time.perf_counter() - start

    # Длительность получившегося аудио = число сэмплов / частота
    with wave.open(out_path, "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()

    # RTF < 1 — синтезируем быстрее, чем длится сама речь
    rtf = elapsed / duration if duration else 0
    print(f"Голос: irina | аудио: {duration:.1f} с | синтез: {elapsed:.2f} с | RTF: {rtf:.3f}")
    return out_path


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else (
        "Привет! Я голосовой ассистент. Сегодня тридцатое сентября, "
        "и я умею превращать текст в речь."
    )
    print(synthesize(text))
