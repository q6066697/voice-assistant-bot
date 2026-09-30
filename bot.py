"""Telegram-бот: голосовой вопрос -> голосовой ответ (локальные модели)."""
import asyncio
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Нет TELEGRAM_BOT_TOKEN в .env")

# Импорт загружает модели Whisper и Piper один раз при старте бота
from guard import clarification  # noqa: E402
from llm import answer  # noqa: E402
from pipeline import speech_to_text, text_to_speech  # noqa: E402

bot = Bot(TOKEN)
dp = Dispatcher()
# На 4 ядрах CPU параллельная обработка только замедлит всё — обрабатываем по очереди
processing_lock = asyncio.Lock()


def wav_to_ogg(wav_path: str, ogg_path: str) -> None:
    """Telegram принимает голосовые только в OGG/Opus, Piper выдаёт WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
         "-c:a", "libopus", "-b:a", "32k", ogg_path],
        check=True,
    )


def process_voice(in_path: str, workdir: str) -> tuple[str, str, str, str]:
    """Голос -> текст -> ответ -> голос. Работает синхронно, вызывается в отдельном потоке."""
    t0 = time.perf_counter()
    question = speech_to_text(in_path)
    t1 = time.perf_counter()
    reply = clarification(question) or answer(question)
    t2 = time.perf_counter()
    wav_path = str(Path(workdir) / "answer.wav")
    ogg_path = str(Path(workdir) / "answer.ogg")
    text_to_speech(reply, wav_path)
    wav_to_ogg(wav_path, ogg_path)
    t3 = time.perf_counter()
    timings = f"STT {t1 - t0:.0f} с · LLM {t2 - t1:.0f} с · TTS {t3 - t2:.0f} с"
    return question, reply, ogg_path, timings


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Привет! Отправь голосовое с вопросом про AI или программирование — отвечу голосом.\n"
        "Работаю на локальных моделях без облака, ответ займёт около 30 секунд."
    )


@dp.message(F.voice)
async def on_voice(message: Message) -> None:
    status = await message.answer("🎧 Слушаю и думаю...")
    async with processing_lock:
        await bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)
        with tempfile.TemporaryDirectory() as tmp:
            in_path = str(Path(tmp) / "question.ogg")
            await bot.download(message.voice, destination=in_path)
            try:
                question, reply, ogg_path, timings = await asyncio.to_thread(
                    process_voice, in_path, tmp
                )
            except Exception:
                logging.exception("Ошибка обработки голосового")
                await status.edit_text("Не получилось обработать сообщение, попробуй ещё раз.")
                return
            caption = f"Вопрос: {question}\nОтвет: {reply}\n\n{timings}"[:1024]
            await message.answer_voice(FSInputFile(ogg_path), caption=caption)
    await status.delete()


@dp.message(F.text)
async def on_text(message: Message) -> None:
    await message.answer("Я отвечаю на голосовые сообщения — запиши вопрос голосом.")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
