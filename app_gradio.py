import gradio as gr
from video_to_text import transcribe  # вынеси функцию отдельно

def ui(video, model, lang, use_gpu):
    if video is None:
        return "Загрузите видео", None
    txt_path = transcribe(video, model=model, lang=lang, use_gpu=use_gpu)
    return open(txt_path, encoding="utf-8").read(), txt_path

with gr.Interface(
    fn=ui,
    inputs=[
        gr.Video(label="Видео"),
        gr.Dropdown(["tiny","base","small","medium","large-v3"], value="small"),
        gr.Textbox(value="ru", label="Язык"),
        gr.Checkbox(value=True, label="Использовать GPU"),
    ],
    outputs=[gr.Textbox(label="Транскрипт", lines=20), gr.File(label="Скачать .txt")],
    title="Video → Text",
) as demo:
    demo.launch(inbrowser=True)