"""
app_gradio.py — веб-интерфейс для v2t на Gradio.

Запуск:
    python app_gradio.py
    или start_gui.bat
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from v2t import transcribe


# ---------- Константы ----------
MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANGS = ["ru", "en", "auto", "uk", "de", "fr", "es", "it", "pl", "tr"]
DEVICES = ["auto", "cuda", "cpu"]


# ---------- Основная функция-обёртка ----------
def run_transcribe(
    video_path: str | None,
    model_name: str,
    lang: str,
    device: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """
    Вызывается Gradio при нажатии кнопки.
    Возвращает: (текст, путь_к_txt_для_скачивания, статус)
    """
    if not video_path:
        return "", None, "⚠️ Загрузи видеофайл или укажи путь."

    # Пути: работаем во временной папке Gradio, чтобы не мусорить
    video = Path(video_path)
    if not video.exists():
        return "", None, f"❌ Файл не найден: {video}"

    work_dir = Path(tempfile.mkdtemp(prefix="v2t_gui_"))

    # Прогресс-бар: у нас 3 смысловых этапа
    def on_progress(stage: str, info: str) -> None:
        # Gradio ждёт значение от 0 до 1
        mapping = {"extract": 0.15, "model": 0.35, "transcribe": 0.6, "done": 1.0}
        progress(mapping.get(stage, 0.5), desc=info or stage)

    try:
        progress(0.05, desc="Старт...")
        result = transcribe(
            video,
            model=model_name,
            lang=lang,
            device=device,
            out_dir=work_dir,
            keep_audio=False,
            progress=on_progress,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return "", None, f"❌ Ошибка: {e}\n\n{tb}"
    finally:
        # Чистим промежуточный .ogg, если остался
        for f in work_dir.glob("*.ogg"):
            try:
                f.unlink()
            except OSError:
                pass

    # Копируем результат во временный файл с «человеческим» именем,
    # чтобы пользователь скачал lesson_transcript.txt, а не случайный хэш.
    download_path = work_dir / f"{video.stem}_transcript.txt"
    if not download_path.exists():
        download_path = result.transcript_path

    status = (
        f"✅ Готово\n"
        f"Язык: {lang} | Модель: {model_name} | Устройство: {device}\n"
        f"Сегментов: {len(result.segments)} | Символов: {len(result.text)}"
    )
    return result.text, str(download_path), status


# ---------- Сборка интерфейса ----------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="v2t — Video to Text") as demo:
        gr.Markdown(
            "# 🎬 v2t — транскрибация видео в текст\n"
            "Загрузи видео или вставь путь к нему, выбери модель и язык — "
            "получишь текст и `.txt` с таймкодами."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(
                    label="Видео (загрузи файл)",
                    sources=["upload"],
                )
                path_in = gr.Textbox(
                    label="…или путь к видео на диске",
                    placeholder=r"C:\path\to\video.mp4",
                )
                with gr.Row():
                    model_dd = gr.Dropdown(
                        MODELS, value="small", label="Модель",
                        info="tiny→large-v3: больше = точнее, но медленнее",
                    )
                    lang_dd = gr.Dropdown(
                        LANGS, value="ru", label="Язык",
                    )
                    device_dd = gr.Dropdown(
                        DEVICES, value="auto", label="Устройство",
                    )

                run_btn = gr.Button("🚀 Транскрибировать", variant="primary")

            with gr.Column(scale=2):
                text_out = gr.Textbox(
                    label="Транскрипт",
                    lines=22,
                )
                file_out = gr.File(label="📄 Скачать .txt с таймкодами")
                status_out = gr.Markdown("Готов к работе.")

        def pick_source(video_path, manual_path):
            if video_path:
                return video_path
            return manual_path or ""

        run_btn.click(
            fn=lambda v, p, m, l, d: run_transcribe(pick_source(v, p), m, l, d),
            inputs=[video_in, path_in, model_dd, lang_dd, device_dd],
            outputs=[text_out, file_out, status_out],
        )

        gr.Markdown(
            "_Первый запуск с новой моделью может занять до минуты — "
            "она скачивается из интернета в кэш HuggingFace._"
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()