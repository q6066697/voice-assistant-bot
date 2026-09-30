#!/usr/bin/env bash
# Скачивает модели для voice-assistant-bot из официальных источников.
set -euo pipefail

VOICE="ru_RU-irina-medium"     # голос Piper для синтеза речи
WHISPER_MODEL="small"          # модель faster-whisper для распознавания

echo "==> Голос Piper: ${VOICE}"
mkdir -p voices
(cd voices && python -m piper.download_voices "${VOICE}")

echo "==> Модель Whisper: ${WHISPER_MODEL}"
python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

echo "==> Готово"
