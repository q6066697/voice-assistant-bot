"""Распознавание речи из аудиофайла через faster-whisper."""
import sys
import time

from faster_whisper import WhisperModel


def transcribe(path: str, model_size: str = "base", beam_size: int = 5,
               prompt: str | None = None) -> str:
    # int8 на CPU — быстрее и экономнее по памяти, качество почти не страдает
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    start = time.perf_counter()
    segments, info = model.transcribe(
        path, language="ru", vad_filter=True, beam_size=beam_size,
        initial_prompt=prompt,  # подсказка словаря: ожидаемые термины
    )
    text = " ".join(segment.text.strip() for segment in segments)
    elapsed = time.perf_counter() - start

    # RTF (real-time factor): < 1 значит, что распознаём быстрее реального времени
    rtf = elapsed / info.duration if info.duration else 0
    print(f"Модель: {model_size} | beam: {beam_size} | prompt: {'да' if prompt else 'нет'} | "
          f"аудио: {info.duration:.1f} с | обработка: {elapsed:.1f} с | RTF: {rtf:.2f}")
    return text


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "samples/test.ogg"
    size = sys.argv[2] if len(sys.argv) > 2 else "base"
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    prompt = sys.argv[4] if len(sys.argv) > 4 else None
    print(transcribe(audio, size, beam, prompt))
