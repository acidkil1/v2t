"""
app_gradio.py — веб-интерфейс для v2t на Gradio.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

from v2t import transcribe


MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANGS = [
    ("Русский", "ru"),
    ("English", "en"),
    ("Auto-detect", "auto"),
    ("Українська", "uk"),
    ("Deutsch", "de"),
    ("Français", "fr"),
    ("Español", "es"),
    ("Italiano", "it"),
    ("Polski", "pl"),
    ("Türkçe", "tr"),
]
DEVICES = ["auto", "cuda", "cpu"]


CSS = """
.gradio-container { max-width: 1400px !important; margin: auto !important; }

#header { text-align: center; margin-bottom: 12px; }
#header h1 { margin-bottom: 4px; }
#header p  { opacity: 0.7; margin-top: 0; }

#left-col, #right-col {
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px !important;
}

#run-btn {
    font-size: 1.05em !important;
    font-weight: 600 !important;
    padding: 12px !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35) !important;
}

#cancel-btn {
    font-size: 0.95em !important;
    padding: 10px !important;
}

input:focus, textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
}

#status-box { line-height: 1.5; }

/* ===== Полоса прогресса ===== */
#progress-wrap {
    margin: 8px 0 4px 0;
    padding: 12px 14px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
}

#progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9em;
    opacity: 0.9;
    margin-bottom: 6px;
    font-weight: 500;
}

#progress-timer {
    font-family: ui-monospace, monospace;
    opacity: 0.7;
}

#progress-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(139, 92, 246, 0.15);
    border-radius: 6px;
    overflow: hidden;
}

#progress-bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #8b5cf6, #a78bfa);
    border-radius: 6px;
    transition: width 0.25s ease;
}
"""


def render_progress(pct: int, label: str, elapsed: float = None) -> str:
    pct = max(0, min(100, int(pct)))
    timer_html = ""
    if elapsed is not None:
        m, s = divmod(int(elapsed), 60)
        timer_html = f"<span id='progress-timer'>{m:02d}:{s:02d}</span>"
    return f"""
    <div id="progress-wrap">
        <div id="progress-label">
            <span>{label}</span>
            {timer_html}
            <span>{pct}%</span>
        </div>
        <div id="progress-bar-bg">
            <div id="progress-bar-fill" style="width: {pct}%;"></div>
        </div>
    </div>
    """


def render_progress_idle() -> str:
    return render_progress(0, "Ожидание")


# ---------- Основной генератор ----------
def run_transcribe(
    video_path,
    model_name: str,
    lang: str,
    device: str,
    save_to_source: bool,
    cancel_flag: dict,
):
    """Генератор. Отдаёт 4 значения: (текст, файл, статус, HTML-полоса)."""
    t0 = time.time()
    cancel_flag["cancelled"] = False

    def pack(text, file, status, pct, label, elapsed=None):
        return text, file, status, render_progress(pct, label, elapsed)

    # --- Проверки ---
    if not video_path:
        yield pack("", None, "⚠️ Загрузи видеофайл или укажи путь.", 0, "Ожидание")
        return

    if isinstance(video_path, Path):
        src = video_path
    else:
        src = Path(str(video_path).strip().strip('"'))

    if not src.exists():
        yield pack("", None, f"❌ Файл не найден: `{src}`", 0, "Ошибка")
        return

    yield pack("", None, "⏳ Готовлю файл…", 3, "Подготовка", 0)

    work_dir = Path(tempfile.mkdtemp(prefix="v2t_gui_"))
    local_video = work_dir / src.name
    try:
        shutil.copy2(src, local_video)
    except (OSError, PermissionError) as e:
        yield pack("", None, f"❌ Не удалось скопировать файл: {e}", 0, "Ошибка", 0)
        return

    if cancel_flag.get("cancelled"):
        yield pack("", None, "🛑 Отменено пользователем.", 0, "Отменено", time.time() - t0)
        return

    out_dir = src.parent if save_to_source else work_dir

    # --- Разделяемое состояние между потоком и генератором ---
    state = {
        "pct": 5,
        "label": "Старт",
        "result": None,
        "error": None,
        "done": False,
    }

    STAGE_MAP = {
        "extract":    (15, "🎵 Извлечение аудио"),
        "download":   (40, "📦 Загрузка модели"),
        "model":      (55, "🧠 Инициализация модели"),
        "transcribe": (70, "💬 Распознавание речи"),
        "done":       (100, "✅ Завершено"),
    }

    def worker():
        def on_progress(stage: str, info: str) -> None:
            pct, label = STAGE_MAP.get(stage, (50, info or stage))
            state["pct"] = pct
            state["label"] = label

        try:
            result = transcribe(
                local_video,
                model=model_name,
                lang=lang,
                device=device,
                out_dir=out_dir,
                keep_audio=False,
                progress=on_progress,
            )
            state["result"] = result
        except Exception as e:
            state["error"] = (type(e).__name__, str(e), traceback.format_exc())
        finally:
            state["done"] = True

    # Запускаем транскрибацию в фоне
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    # --- Основной цикл: обновляем UI, пока поток работает ---
    last_pct = -1
    last_label = ""

    while not state["done"]:
        elapsed = time.time() - t0

        # Отмену проверяем на каждом тике
        if cancel_flag.get("cancelled"):
            yield pack("", None, "🛑 Отменено пользователем.", state["pct"], "Отменено", elapsed)
            return

        # Обновляем UI только когда реально что-то изменилось,
        # чтобы не спамить Gradio (иначе лагает)
        cur_pct = state["pct"]
        cur_label = state["label"]

        if cur_pct != last_pct or cur_label != last_label:
            last_pct = cur_pct
            last_label = cur_label

        # Yield каждые ~0.4 сек для таймера
        yield pack(
            "", None,
            f"⏳ {cur_label}…",
            cur_pct, cur_label, elapsed,
        )
        time.sleep(0.4)

    th.join(timeout=1)
    elapsed_total = time.time() - t0

    # --- Обработка ошибки из потока ---
    if state["error"]:
        etype, emsg, etb = state["error"]
        yield pack(
            "",
            None,
            f"❌ **{etype}:** {emsg}\n\n```\n{etb}\n```",
            0, "Ошибка", elapsed_total,
        )
        return

    # --- Финализация ---
    result = state["result"]
    if result is None:
        yield pack("", None, "❌ Неизвестная ошибка: результат пуст.", 0, "Ошибка", elapsed_total)
        return

    final_txt = out_dir / f"{src.stem}_transcript.txt"
    if result.transcript_path != final_txt:
        try:
            shutil.copy2(result.transcript_path, final_txt)
        except OSError:
            final_txt = result.transcript_path

    status = (
        f"### ✅ Готово за {elapsed_total:.1f} сек\n\n"
        f"| Параметр | Значение |\n"
        f"|---|---|\n"
        f"| Модель | `{model_name}` |\n"
        f"| Язык | `{lang}` |\n"
        f"| Устройство | `{device}` |\n"
        f"| Сегментов | `{len(result.segments)}` |\n"
        f"| Символов | `{len(result.text)}` |\n"
        f"| Файл | `{final_txt.name}` |\n"
    )

    yield pack(result.text, str(final_txt), status, 100, "Завершено", elapsed_total)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="v2t — Video to Text") as demo:
        cancel_flag = gr.State({"cancelled": False})

        with gr.Column(elem_id="header"):
            gr.Markdown(
                "# 🎬 v2t — транскрибация видео в текст\n"
                "Загрузи видео, выбери пресет — получишь текст и `.txt` с таймкодами."
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="left-col"):
                gr.Markdown("### 📥 Источник")

                video_in = gr.File(
                    label="Видео или аудио (drag & drop или клик)",
                    file_types=["video", "audio"],
                    type="filepath",
                )

                path_in = gr.Textbox(
                    label="…или путь к файлу на диске",
                    placeholder=r"C:\path\to\video.mp4",
                    lines=1,
                )

                gr.Markdown("### ⚙️ Настройки")

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

                with gr.Row():
                    run_btn = gr.Button(
                        "🚀 Транскрибировать",
                        variant="primary",
                        elem_id="run-btn",
                        scale=3,
                    )
                    cancel_btn = gr.Button(
                        "🛑 Отменить",
                        variant="secondary",
                        elem_id="cancel-btn",
                        scale=1,
                    )

            with gr.Column(scale=7, elem_id="right-col"):
                gr.Markdown("### 📝 Результат")

                progress_out = gr.HTML(render_progress_idle())

                text_out = gr.Textbox(
                    label=None,
                    show_label=False,
                    lines=16,
                    placeholder="Здесь появится распознанный текст…",
                )

                file_out = gr.File(label="📄 Скачать .txt")
                status_out = gr.Markdown(
                    "_Готов к работе. Загрузи видео слева и нажми «Транскрибировать»._",
                    elem_id="status-box",
                )

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

        def launch_transcribe(v, p, m, l, d, s, c):
            yield from run_transcribe(pick_source(v, p), m, l, d, s, c)

        run_evt = run_btn.click(
            fn=launch_transcribe,
            inputs=[video_in, path_in, model_dd, lang_dd, device_dd, save_to_source, cancel_flag],
            outputs=[text_out, file_out, status_out, progress_out],
        )

        def do_cancel(flag):
            flag["cancelled"] = True
            return flag, "🛑 **Запрошена отмена.** Дождись текущего шага."

        cancel_btn.click(
            fn=do_cancel,
            inputs=[cancel_flag],
            outputs=[cancel_flag, status_out],
            cancels=[run_evt],
        )

        gr.Markdown(
            "<div style='text-align:center; opacity:0.6; font-size:0.85em;'>"
            "Первый запуск с новой моделью может занять до минуты — "
            "она качается из интернета и кэшируется. Повторные запуски — мгновенно."
            "</div>"
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()