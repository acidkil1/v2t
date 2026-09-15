"""
app_gradio.py — веб-интерфейс для v2t на Gradio.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr

from v2t import transcribe


MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANGS = ["ru", "en", "auto", "uk", "de", "fr", "es", "it", "pl", "tr"]
DEVICES = ["auto", "cuda", "cpu"]


def run_transcribe(
    video_path,
    model_name: str,
    lang: str,
    device: str,
    save_to_source: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Возвращает: (текст, путь_к_txt, статус, время)"""
    t0 = time.time()

    if not video_path:
        return "", None, "⚠️ Загрузи видеофайл или укажи путь.", ""

    # Gradio может отдать str или Path
    if isinstance(video_path, Path):
        video = video_path
    else:
        video = Path(str(video_path).strip().strip('"'))

    if not video.exists():
        return "", None, f"❌ Файл не найден: {video}", ""

    if save_to_source:
        work_dir = video.parent
        cleanup_after = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="v2t_gui_"))
        cleanup_after = True

    STAGE_PCT = {
        "extract": 0.10,
        "download": 0.30,
        "model": 0.45,
        "transcribe": 0.65,
        "done": 1.00,
    }

    def on_progress(stage: str, info: str) -> None:
        progress(STAGE_PCT.get(stage, 0.5), desc=info or stage)

    try:
        progress(0.02, desc="Старт...")
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
        elapsed = time.time() - t0
        return (
            "",
            None,
            f"❌ **{type(e).__name__}:** {e}\n\n```\n{tb}\n```",
            f"⏱ {elapsed:.1f} сек (с ошибкой)",
        )

    final_txt = work_dir / f"{video.stem}_transcript.txt"
    if result.transcript_path != final_txt:
        try:
            shutil.copy2(result.transcript_path, final_txt)
        except OSError:
            final_txt = result.transcript_path

    elapsed = time.time() - t0
    elapsed_str = f"⏱ {elapsed:.1f} сек"

    status = (
        f"✅ **Готово**\n\n"
        f"- Модель: `{model_name}`\n"
        f"- Язык: `{lang}`\n"
        f"- Устройство: `{device}`\n"
        f"- Сегментов: `{len(result.segments)}`\n"
        f"- Символов: `{len(result.text)}`\n"
        f"- Время: `{elapsed:.1f} сек`"
    )

    if cleanup_after:
        cache_dir = Path(tempfile.gettempdir()) / "v2t_results"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / final_txt.name
        try:
            shutil.copy2(final_txt, cached)
            final_txt = cached
        except OSError:
            pass

    return result.text, str(final_txt), status, elapsed_str


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="v2t — Video to Text") as demo:
        gr.Markdown(
            "# 🎬 v2t — транскрибация видео в текст\n"
            "Загрузи видеофайл (или вставь путь), выбери пресет — получишь текст и `.txt` с таймкодами."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.File(
                    label="Видео (drag & drop или клик)",
                    file_types=["video", "audio"],
                    type="filepath",
                )
                path_in = gr.Textbox(
                    label="…или путь к файлу на диске",
                    placeholder=r"C:\path\to\video.mp4",
                )

                preset = gr.Radio(
                    choices=[
                        ("⚡ Быстро (tiny, CPU)", "fast"),
                        ("⚖️ Сбалансированно (small, auto)", "balanced"),
                        ("🎯 Качественно (medium, auto)", "quality"),
                        ("🔧 Вручную", "custom"),
                    ],
                    value="balanced",
                    label="Пресет",
                )

                with gr.Group(visible=False) as manual_group:
                    with gr.Row():
                        model_dd = gr.Dropdown(MODELS, value="small", label="Модель")
                        lang_dd = gr.Dropdown(LANGS, value="ru", label="Язык")
                        device_dd = gr.Dropdown(DEVICES, value="auto", label="Устройство")

                save_to_source = gr.Checkbox(
                    value=True,
                    label="Сохранять .txt рядом с видео (иначе — во временную папку)",
                )

                run_btn = gr.Button("🚀 Транскрибировать", variant="primary")

            with gr.Column(scale=2):
                text_out = gr.Textbox(label="Транскрипт", lines=20)
                file_out = gr.File(label="📄 Скачать .txt с таймкодами")
                elapsed_out = gr.Markdown("")
                status_out = gr.Markdown("Готов к работе.")

        def apply_preset(p):
            if p == "fast":
                return (
                    gr.update(value="tiny", interactive=False),
                    gr.update(value="ru", interactive=False),
                    gr.update(value="cpu", interactive=False),
                    gr.update(visible=False),
                )
            if p == "balanced":
                return (
                    gr.update(value="small", interactive=False),
                    gr.update(value="ru", interactive=False),
                    gr.update(value="auto", interactive=False),
                    gr.update(visible=False),
                )
            if p == "quality":
                return (
                    gr.update(value="medium", interactive=False),
                    gr.update(value="ru", interactive=False),
                    gr.update(value="auto", interactive=False),
                    gr.update(visible=False),
                )
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(visible=True),
            )

        preset.change(
            fn=apply_preset,
            inputs=[preset],
            outputs=[model_dd, lang_dd, device_dd, manual_group],
        )

        def pick_source(video_path, manual_path):
            if video_path:
                return video_path
            return manual_path or ""

        run_btn.click(
            fn=lambda v, p, m, l, d, s: run_transcribe(pick_source(v, p), m, l, d, s),
            inputs=[video_in, path_in, model_dd, lang_dd, device_dd, save_to_source],
            outputs=[text_out, file_out, status_out, elapsed_out],
        )

        gr.Markdown(
            "_Первый запуск с новой моделью может занять до минуты — "
            "она качается из интернета и кэшируется. Повторные запуски — мгновенно._"
        )

    return demo

def main() -> None:
    demo = build_ui()
    demo.queue().launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()