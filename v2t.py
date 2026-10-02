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
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

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
    Работает молча — никакого вывода в консоль.
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
        for bd in bin_dirs:
            for dll in bd.glob("*.dll"):
                target = scripts_dir / dll.name
                if not target.exists() or target.stat().st_size != dll.stat().st_size:
                    try:
                        shutil.copy2(dll, target)
                    except Exception:
                        pass
    except Exception:
        pass

    # --- 4. Предзагружаем ключевые DLL ---
    try:
        import ctypes
        for bd in bin_dirs:
            for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"):
                f = bd / name
                if f.exists():
                    try:
                        ctypes.WinDLL(str(f))
                    except OSError:
                        continue
    except Exception:
        pass


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


# ---------- Скачивание по ссылке ----------
def _is_url(s: str) -> bool:
    """True, если строка начинается с http:// или https:// (значит, это ссылка)."""
    return isinstance(s, str) and s.strip().lower().startswith(("http://", "https://"))


def _filename_from_href(href: str) -> str | None:
    """Пытается вытащить имя файла из временной ссылки Яндекс.Диска (?filename=…)."""
    try:
        query = parse_qs(urlparse(href).query)
        name = query.get("filename", [None])[0]
        if name:
            return Path(name).name
    except Exception:
        pass
    return None


def _download_from_yandex_disk(url: str, target_dir: Path, report) -> Path | None:
    """
    Прямое скачивание публичного файла с Яндекс.Диска через официальный API.
    Возвращает Path к скачанному файлу либо None, если ссылка не про Яндекс.Диск
    (тогда вызывающий код переключится на yt-dlp). Ошибки API пробрасываются
    наверх как RuntimeError, чтобы сработал fallback.
    """
    host = (urlparse(url).netloc or "").lower()
    # Работаем с API только для ссылок Яндекса — для остальных сразу yt-dlp
    if "yandex" not in host and not host.endswith("yadi.sk"):
        return None

    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    resp = requests.get(f"{api}?{urlencode({'public_key': url})}", timeout=15)
    resp.raise_for_status()

    data = resp.json()
    href = data.get("href")
    if not href:
        raise RuntimeError("API Яндекс.Диска не вернул ссылку на скачивание (возможно, это папка).")

    # Имя файла: сначала из временной ссылки, иначе — общий дефолт
    filename = _filename_from_href(href) or "yandex_disk_file.mp4"
    dest = target_dir / filename

    with requests.get(href, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    report("download_url", f"Скачивание… {int(done * 100 / total)}%")
    return dest


def _download_with_ytdlp(url: str, target_dir: Path, report) -> Path:
    """Fallback-скачивание через yt-dlp (Яндекс.Диск, YouTube, VK Video)."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp не установлен. Установите зависимости: pip install -r requirements.txt"
        )

    report("download_url", "Скачивание через yt-dlp…")

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                report("download_url", f"Скачивание… {int(done * 100 / total)}%")

    opts = {
        "outtmpl": str(target_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        try:
            path = Path(ydl.prepare_filename(info))
        except Exception:
            # Плейлист/папка: у info нет единого имени файла
            path = None

    if path is not None and path.exists():
        return path

    # Иногда расширение в имени не совпадает с фактическим файлом —
    # берём самый свежий файл в целевой папке.
    candidates = sorted(
        (p for p in target_dir.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise RuntimeError("yt-dlp не смог скачать файл по ссылке.")


def download_from_url(url: str, target_dir: Path, progress=None) -> Path:
    """
    Скачивает видео по ссылке. Сначала пробует прямой API Яндекс.Диска,
    при неудаче — yt-dlp. Возвращает Path к скачанному файлу.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, info: str = "") -> None:
        if progress is not None:
            try:
                progress(stage, info)
            except Exception:
                pass

    report("download_url", "Скачивание видео…")

    # 1) Прямой API Яндекс.Диска
    try:
        downloaded = _download_from_yandex_disk(url, target_dir, report)
        if downloaded is not None:
            report("download_url", "Скачано")
            return downloaded
    except Exception as e:
        report("download_url", f"API Яндекс.Диска недоступен ({e}), пробую yt-dlp…")

    # 2) Fallback: yt-dlp (Яндекс.Диск, YouTube, VK Video)
    downloaded = _download_with_ytdlp(url, target_dir, report)
    report("download_url", "Скачано")
    return downloaded


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
    keep_audio: bool = False,
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
    raw = video if isinstance(video, str) else str(video)
    raw = raw.strip().strip('"')

    # Источник: ссылка (URL) или локальный файл
    url_source = _is_url(raw)
    temp_dir: Path | None = None

    if url_source:
        temp_dir = Path(tempfile.mkdtemp(prefix="v2t_url_"))
        try:
            video_path = download_from_url(raw, temp_dir, progress=report)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    else:
        video_path = Path(raw).resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Файл не найден: {video_path}")

        if video_path.suffix.lower() not in (VIDEO_EXTS | AUDIO_EXTS):
            raise ValueError(
                f"Похоже, это не видео/аудио: {video_path.suffix}. "
                f"Поддерживаются: {', '.join(sorted(VIDEO_EXTS | AUDIO_EXTS))}"
            )

    try:
        # Куда писать промежуточные файлы
        if out_dir is None:
            # Для ссылки «рядом с видео» нет — пишем в текущую папку
            work_dir = Path.cwd() if url_source else video_path.parent
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
    finally:
        # Удаляем временную папку со скачанным по ссылке файлом
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------- CLI ----------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v2t")
    parser.add_argument("video", help="путь к видеофайлу или URL (Яндекс.Диск, YouTube, VK)")
    parser.add_argument("--model", default="small",
                        help="модель: tiny/base/small/medium/large-v3 (по умолчанию small)")
    parser.add_argument("--lang", default="ru",
                        help="язык: ru/en/... или auto (по умолчанию ru)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="устройство (по умолчанию auto)")
    parser.add_argument("--out", default=None,
                        help="папка для .ogg и _transcript.txt (по умолчанию рядом с видео)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="сохранять промежуточный .ogg (по умолчанию удаляется)")

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
            keep_audio=args.keep_audio,
            progress=cli_progress,
        )
    except Exception as e:
        print(f"\n[ОШИБКА] {e}", file=sys.stderr)
        return 1

    print(f"\n[ГОТОВО] {result.transcript_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())