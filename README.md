# v2t — Video to Text

Транскрибация видео в текст через [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Установка

1. Установи [Python 3.10 или 3.11](https://www.python.org/downloads/) (галочка «Add Python to PATH» обязательна).
2. Скачай [ffmpeg essentials build](https://www.gyan.dev/ffmpeg/builds/) и положи `ffmpeg.exe` в `ffmpeg/bin/`.
3. Запусти `install.bat`.

## Использование
``` v2t.bat "C:\путь\к\видео.mp4" ```

Опции:
- `--model tiny|base|small|medium|large-v3` (по умолчанию `small`)
- `--lang ru|en|...|auto` (по умолчанию `ru`)
- `--device auto|cuda|cpu` (по умолчанию `auto`)
- `--out ПАПКА` — куда писать результат
- `--delete-audio` — удалить промежуточный .ogg