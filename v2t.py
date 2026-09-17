"""
v2t.py — транскрибация видео в текст через faster-whisper.
"""

from __future__ import annotations

import argparse
import os
import shutil
import site
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# === Отключаем шумные предупреждения HuggingFace ===
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from faster_whisper import WhisperModel


# ---------- Регистрация NVIDIA DLL (для CUDA) ----------
def _register_nvidia_dll_dirs() -> None:
    """
    Windows-специфичная регистрация NVIDIA DLL из пакетов nvidia-*-cu12.
    Делает три вещи:
      1) os.add_dll_directory — для Python-кода (ctypes и т.п.)
      2) Копирует DLL рядом с python.exe — для C++ кода ctranslate2,
         который не читает os.add_dll_directory.
      3) Предзагружает ключевые DLL в процесс.
    """
    if sys.platform != "win32":
        return

    # --- 1. Собираем все папки nvidia/*/bin ---
    nvidia_roots = []
    try:
        nvidia_roots.append(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia")
        for sp in site.getsitepackages():
            nvidia_roots.append(Path(sp) / "nvidia")
        try:
            nvidia_roots.append(Path(site.getusersitepackages()) / "nvidia")
        except Exception:
            pass
    except Exception:
        pass

    bin_dirs = []
    for root in nvidia_roots:
        if not root.exists():
            continue
        for sub in root.iterdir():
            bd = sub / "bin"
            if bd.exists() and any(bd.glob("*.dll")):
                bin_dirs.append(bd)

    if not bin_dirs:
        return

    # --- 2. os.add_dll_directory ---
    for bd in bin_dirs:
        try:
            os.add_dll_directory(str(bd))
        except Exception:
            pass

    # --- 3. Копируем DLL рядом с python.exe (venv\Scripts) ---
    try:
        scripts_dir = Path(sys.executable).parent
        copied = 0
        for bd in bin_dirs:
            for dll in bd.glob("*.dll"):
                target = scripts_dir / dll.name
                if not target.exists() or target.stat().st_size != dll.stat().st_size:
                    try:
                        shutil.copy2(dll, target)
                        copied += 1
                    except Exception:
                        pass
        if copied:
            print(f"[i] Обновлено {copied} NVIDIA DLL в {scripts_dir}")
    except Exception as e:
        print(f"[!] Не удалось скопировать NVIDIA DLL: {e}", file=sys.stderr)

    # --- 4. Предзагружаем ключевые DLL ---
    try:
        import ctypes
        loaded = 0
        for bd in bin_dirs:
            for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"):
                f = bd / name
                if f.exists():
                    try:
                        ctypes.WinDLL(str(f))
                        loaded += 1
                    except OSError:
                        continue
        if loaded:
            print(f"[i] Предзагружено {loaded} NVIDIA DLL")
    except Exception as e:
        print(f"[!] Не удалось предзагрузить NVIDIA DLL: {e}", file=sys.stderr)


_register_nvidia_dll_dirs()


# ---------- Кэш моделей ----------
_MODEL_CACHE: dict[tuple[str, str], WhisperModel] = {}


def get_model(name: str = "small", device: str = "auto", progress=None) -> WhisperModel:
    """Возвращает WhisperModel. Кэширует по (name, device)."""

    def report(stage: str, info: str = "") -> None:
        if progress:
            try:
                progress(stage, info)
            except Exception:
                pass

    if device == "auto":
        try:
            report("download", f"Подготовка модели {name} (CUDA)...")
            model = WhisperModel(name, device="cuda", compute_type="int8_float16")
            import numpy as np
            dummy = np.zeros(16000, dtype=np.float32)
            list(model.transcribe(dummy, language="ru")[0])
            _MODEL_CACHE[(name, "cuda")] = model
            print("[i] Используется GPU (CUDA)")
            return model
        except Exception as e:
            print(f"[!] CUDA не завелась ({e.__class__.__name__}: {e}), использую CPU",
                  file=sys.stderr)
            report("download", f"Подготовка модели {name} (CPU)...")
            model = WhisperModel(name, device="cpu", compute_type="int8")
            _MODEL_CACHE[(name, "cpu")] = model
            return model

    key = (name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    report("download", f"Подготовка модели {name} ({device})...")
    if device == "cuda":
        model = WhisperModel(name, device="cuda", compute_type="int8_float16")
    else:
        model = WhisperModel(name, device="cpu", compute_type="int8")
    _MODEL_CACHE[key] = model
    return model


# ---------- Извлечение аудио ----------
def _find_ffmpeg() -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "ffmpeg" / "bin" / "ffmpeg.exe",
        here / "ffmpeg" / "bin" / "ffmpeg",
        here / "ffmpeg.exe",
        here / "ffmpeg",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "ffmpeg"


def extract_audio(video: Path, audio: Path) -> Path:
    audio.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    try:
        subprocess.run(
            [
                ffmpeg,
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
            "ffmpeg не найден. Положи ffmpeg.exe в папку ffmpeg\\bin\\ рядом с v2t.py."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg завершился с ошибкой (код {e.returncode}).")
    return audio


# ---------- Результат ----------
@dataclass
class TranscriptionResult:
    text: str
    segments: list[tuple[float, float, str]]
    transcript_path: Path
    audio_path: Path


# ---------- Основная функция ----------
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma"}


def transcribe(
    video: str | Path,
    *,
    model: str = "small",
    lang: str = "ru",
    device: str = "auto",
    out_dir: str | Path | None = None,
    keep_audio: bool = True,
    progress=None,
) -> TranscriptionResult:
    """Транскрибирует видео. Возвращает TranscriptionResult."""

    def report(stage: str, info: str = "") -> None:
        if progress is not None:
            try:
                progress(stage, info)
            except Exception:
                pass

    # Приводим к Path аккуратно (Gradio может отдать и str, и Path)
    if isinstance(video, Path):
        video_path = video.resolve()
    else:
        video_path = Path(str(video).strip().strip('"')).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    if video_path.suffix.lower() not in (VIDEO_EXTS | AUDIO_EXTS):
        raise ValueError(
            f"Похоже, это не видео/аудио: {video_path.suffix}. "
            f"Поддерживаются: {', '.join(sorted(VIDEO_EXTS | AUDIO_EXTS))}"
        )

    # Куда писать промежуточные файлы
    if out_dir is None:
        work_dir = video_path.parent
    else:
        work_dir = Path(out_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / (video_path.stem + ".ogg")
    transcript_path = work_dir / (video_path.stem + "_transcript.txt")

    # Шаг 1: аудио
    report("extract", f"Извлечение аудио из {video_path.name}")
    extract_audio(video_path, audio_path)

    # Шаг 2: модель
    report("download", f"Загрузка модели {model} ({device})")
    m = get_model(model, device, progress=report)

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
    parser.add_argument("--model", default="small")
    parser.add_argument("--lang", default="ru")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--delete-audio", action="store_true")

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
        print(f"\n[ОШИБКА] {e}", file=sys.stderr)
        return 1

    print(f"\n[ГОТОВО] {result.transcript_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())