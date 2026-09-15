"""
video_to_text.py — транскрибация видео в текст через faster-whisper.

Использование как CLI:
    python video_to_text.py "C:\\video.mp4"
    python video_to_text.py "C:\\video.mp4" --model medium --lang en
    python video_to_text.py "C:\\video.mp4" --out "C:\\out"

Использование как библиотеки:
    from video_to_text import transcribe
    result = transcribe("C:/video.mp4", model="small", lang="ru")
    print(result.text)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


# ---------- Кэш моделей ----------
# Чтобы Gradio (или повторные вызовы CLI) не грузили модель заново.
_MODEL_CACHE: dict[tuple[str, str], WhisperModel] = {}


def get_model(name: str = "small", device: str = "auto") -> WhisperModel:
    """
    Возвращает WhisperModel. Кэширует по (name, device).

    device:
      - "auto" — пробуем cuda, при ошибке падаем на cpu
      - "cuda" — только GPU
      - "cpu"  — только CPU
    """
    if device == "auto":
        # Разрешаем auto в конкретное значение и запоминаем результат
        key_cuda = (name, "cuda")
        key_cpu = (name, "cpu")
        if key_cuda in _MODEL_CACHE:
            return _MODEL_CACHE[key_cuda]
        if key_cpu in _MODEL_CACHE:
            return _MODEL_CACHE[key_cpu]
        try:
            model = WhisperModel(name, device="cuda", compute_type="int8_float16")
            _MODEL_CACHE[key_cuda] = model
            return model
        except Exception as e:
            print(f"[!] CUDA недоступна ({e.__class__.__name__}), использую CPU", file=sys.stderr)
            model = WhisperModel(name, device="cpu", compute_type="int8")
            _MODEL_CACHE[key_cpu] = model
            return model

    key = (name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if device == "cuda":
        model = WhisperModel(name, device="cuda", compute_type="int8_float16")
    else:
        model = WhisperModel(name, device="cpu", compute_type="int8")

    _MODEL_CACHE[key] = model
    return model


# ---------- Извлечение аудио ----------
def extract_audio(video: Path, audio: Path) -> Path:
    """
    Извлекает аудио из видео в .ogg (opus, mono, 12k).
    Бросает RuntimeError, если ffmpeg не найден или упал.
    """
    audio.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-stats",
                "-i", str(video),
                "-vn",
                "-acodec", "libopus",
                "-ac", "1",
                "-ab", "12k",
                "-application", "voip",
                "-map_metadata", "-1",
                str(audio),
                "-y",
            ],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установи ffmpeg или положи ffmpeg.exe рядом со скриптом."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg завершился с ошибкой (код {e.returncode}).")
    return audio


# ---------- Результат ----------
@dataclass
class TranscriptionResult:
    text: str                 # весь текст одним куском
    segments: list[tuple[float, float, str]]  # (start, end, text)
    transcript_path: Path     # путь к .txt с таймкодами
    audio_path: Path          # путь к промежуточному .ogg


# ---------- Основная функция ----------
def transcribe(
    video: str | Path,
    *,
    model: str = "small",
    lang: str = "ru",
    device: str = "auto",
    out_dir: str | Path | None = None,
    keep_audio: bool = True,
    progress: callable | None = None,
) -> TranscriptionResult:
    """
    Транскрибирует видео. Возвращает TranscriptionResult.

    :param video: путь к видеофайлу
    :param model: имя модели (tiny/base/small/medium/large-v3)
    :param lang: код языка ("ru", "en", ...) или "auto"
    :param device: "auto" | "cuda" | "cpu"
    :param out_dir: куда писать .ogg и _transcript.txt (по умолчанию — temp)
    :param keep_audio: удалять ли .ogg после обработки
    :param progress: колбэк (stage: str, info: str) для обновления UI
    :return: TranscriptionResult
    """
    def report(stage: str, info: str = "") -> None:
        if progress is not None:
            try:
                progress(stage, info)
            except Exception:
                pass

    video_path = Path(video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    # Куда писать промежуточные файлы
    if out_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="v2t_"))
    else:
        work_dir = Path(out_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / (video_path.stem + ".ogg")
    transcript_path = work_dir / (video_path.stem + "_transcript.txt")

    # Шаг 1: аудио
    report("extract", f"Извлечение аудио из {video_path.name}")
    extract_audio(video_path, audio_path)

    # Шаг 2: модель
    report("model", f"Загрузка модели {model} ({device})")
    m = get_model(model, device)

    # Шаг 3: транскрибация
    report("transcribe", "Распознавание речи...")
    language = None if lang == "auto" else lang
    segments_iter, info = m.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    segments: list[tuple[float, float, str]] = []
    lines: list[str] = []
    for seg in segments_iter:
        text = seg.text.strip()
        segments.append((seg.start, seg.end, text))
        lines.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {text}")

    full_text = " ".join(s[2] for s in segments).strip()

    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not keep_audio and audio_path.exists():
        try:
            audio_path.unlink()
        except OSError:
            pass

    report("done", str(transcript_path))

    return TranscriptionResult(
        text=full_text,
        segments=segments,
        transcript_path=transcript_path,
        audio_path=audio_path,
    )


# ---------- CLI ----------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2t",
        description="Транскрибация видео в текст (faster-whisper).",
    )
    parser.add_argument("video", help="путь к видеофайлу")
    parser.add_argument("--model", default="small",
                        help="модель: tiny/base/small/medium/large-v3 (по умолчанию small)")
    parser.add_argument("--lang", default="ru",
                        help="язык: ru/en/... или auto (по умолчанию ru)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="устройство (по умолчанию auto)")
    parser.add_argument("--out", default=None,
                        help="папка для .ogg и _transcript.txt (по умолчанию рядом с видео)")
    parser.add_argument("--delete-audio", action="store_true",
                        help="удалить промежуточный .ogg после обработки")

    args = parser.parse_args(argv)

    def cli_progress(stage: str, info: str) -> None:
        print(f"[{stage}] {info}")

    try:
        result = transcribe(
            args.video,
            model=args.model,
            lang=args.lang,
            device=args.device,
            out_dir=args.out,
            keep_audio=not args.delete_audio,
            progress=cli_progress,
        )
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        return 1

    print(f"\n✅ Готово: {result.transcript_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())