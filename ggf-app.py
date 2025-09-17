import json
import os
import sys
import threading
import time
import warnings
from uuid import uuid4
import random
import glob

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")

cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto
import yt_dlp
from pydub import AudioSegment

# dirs
os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("voices", exist_ok=True)
os.makedirs("session_tmp", exist_ok=True)

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}

EMO_CHOICES = [i18n("与音色参考音频相同"), i18n("使用情感参考音频"), i18n("使用情感向量控制"), i18n("使用情感描述文本控制")]
EMO_CHOICES_BASE = EMO_CHOICES[:3]  # 基础选项
EMO_CHOICES_EXPERIMENTAL = EMO_CHOICES  # 全部选项（包括文本描述）

MAX_LENGTH_TO_USE_SPEED = 70

with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio", None):
            emo_audio_path = os.path.join("examples", example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([
            os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
            EMO_CHOICES[example.get("emo_mode", 0)],
            example.get("text"),
            emo_audio_path,
            example.get("emo_weight", 1.0),
            example.get("emo_text", ""),
            example.get("emo_vec_1", 0),
            example.get("emo_vec_2", 0),
            example.get("emo_vec_3", 0),
            example.get("emo_vec_4", 0),
            example.get("emo_vec_5", 0),
            example.get("emo_vec_6", 0),
            example.get("emo_vec_7", 0),
            example.get("emo_vec_8", 0),
            example.get("emo_text") is not None
        ])

def normalize_emo_vec(emo_vec):
    # emotion factors for better user experience
    k_vec = [0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45]
    tmp = np.array(k_vec) * np.array(emo_vec)
    if np.sum(tmp) > 0.8:
        tmp = tmp * 0.8 / np.sum(tmp)
    return tmp.tolist()

def _history_add(history_list, label, path):
    # history_list is a list of dicts [{'label': ..., 'path': ...}]
    new_list = list(history_list or [])
    new_list.append({"label": label, "path": path})
    choices = [h["label"] for h in new_list]
    return new_list, choices

def gen_single(
    emo_control_method, prompt, text, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random, max_text_tokens_per_segment,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
    auto_save, history_state, progress=gr.Progress()
):
    # choose output location
    base_dir = "outputs" if auto_save else "session_tmp"
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, f"spk_{uuid4().hex}.wav")

    # set gradio progress
    tts.gr_progress = progress

    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value

    if emo_control_method == 0:
        emo_ref_path = None
    if emo_control_method == 1:
        emo_weight = emo_weight * 0.8
        vec = None
    elif emo_control_method == 2:
        vec = normalize_emo_vec([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")

    output = tts.infer(
        spk_audio_prompt=prompt,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_ref_path,
        emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method == 3),
        emo_text=emo_text,
        use_random=emo_random,
        verbose=cmd_args.verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **kwargs
    )

    # update session history
    label_text = (text or "")[:32].replace("\n", " ")
    label = f"{time.strftime('%H:%M:%S')} | {label_text}" if label_text else f"{time.strftime('%H:%M:%S')} | <no text>"
    updated_history, choices = _history_add(history_state, label, output)
    return (
        gr.update(value=output, visible=True),
        gr.update(choices=choices, value=label, interactive=True),
        updated_history
    )

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def download_yt_audio(url, save_clip_to_voices):
    # returns: prompt_audio, full_audio_preview, start_time, end_time, segment_preview, trim_button, full_path_hidden, refine_row
    if not url:
        return (
            gr.update(value=None),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = glob.glob('temp.*')[0]
        audio = AudioSegment.from_file(audio_file)
        duration_ms = len(audio)
        full_path = 'temp_full.wav'
        audio.export(full_path, format="wav")

        duration_sec = duration_ms / 1000.0
        max_start = max(0, duration_sec - 8)
        start_sec = random.uniform(0, max_start)
        start_ms = int(start_sec * 1000)
        end_sec = min(start_sec + 8, duration_sec)
        end_ms = int(end_sec * 1000)

        excerpt = audio[start_ms:end_ms]
        # put into prompts for immediate use
        prompt_out = os.path.join("prompts", f"yt_{int(time.time())}.wav")
        excerpt.export(prompt_out, format="wav")

        # optional save to voices
        if save_clip_to_voices:
            voice_out = os.path.join("voices", f"voice_yt_{int(time.time())}.wav")
            excerpt.export(voice_out, format="wav")

        # quick preview file
        temp_seg = 'temp_segment.wav'
        excerpt.export(temp_seg, format="wav")

        os.remove(audio_file)
        return (
            gr.update(value=prompt_out),
            gr.update(value=full_path, visible=True),
            gr.update(value=start_sec),
            gr.update(value=end_sec),
            gr.update(value=temp_seg, visible=True),
            gr.update(visible=True),
            gr.update(value=full_path),
            gr.update(visible=True),
        )
    except Exception as e:
        print(f"Error downloading or processing: {e}")
        return (
            gr.update(value=None),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

def preview_segment(start_sec, end_sec, full_path):
    if not full_path or not os.path.exists(full_path):
        return gr.update(value=None, visible=False)
    try:
        audio = AudioSegment.from_file(full_path)
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000.0
        start_ms = int(max(0, start_sec) * 1000)
        end_ms = int(min(duration_sec, end_sec) * 1000)
        if end_ms <= start_ms:
            return gr.update(value=None, visible=False)
        excerpt = audio[start_ms:end_ms]
        temp_seg = 'temp_segment.wav'
        excerpt.export(temp_seg, format="wav")
        return gr.update(value=temp_seg, visible=True)
    except Exception as e:
        print(f"Error previewing segment: {e}")
        return gr.update(value=None, visible=False)

def trim_and_load(start_sec, end_sec, full_path, save_clip_to_voices):
    if not full_path or not os.path.exists(full_path):
        return gr.update(value=None)
    try:
        audio = AudioSegment.from_file(full_path)
        duration_ms = len(audio)
        start_ms = int(max(0, start_sec) * 1000)
        end_ms = int(min(duration_ms / 1000, end_sec) * 1000)
        if end_ms <= start_ms:
            return gr.update(value=None)
        excerpt = audio[start_ms:end_ms]
        output_path = os.path.join("prompts", f"yt_trim_{int(time.time())}.wav")
        excerpt.export(output_path, format="wav")
        if save_clip_to_voices:
            voice_out = os.path.join("voices", f"voice_trim_{int(time.time())}.wav")
            excerpt.export(voice_out, format="wav")
        return gr.update(value=output_path)
    except Exception as e:
        print(f"Error trimming: {e}")
        return gr.update(value=None)

def on_input_text_change(text, max_text_tokens_per_segment):
    if text and len(text) > 0:
        text_tokens_list = tts.tokenizer.tokenize(text)
        segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
        data = []
        for i, s in enumerate(segments):
            segment_str = ''.join(s)
            tokens_count = len(s)
            data.append([i, segment_str, tokens_count])
        return {
            segments_preview: gr.update(value=data, visible=True, type="array"),
        }
    else:
        df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
        return {
            segments_preview: gr.update(value=df),
        }

def on_method_select(emo_control_method):
    if emo_control_method == 1:  # emotion reference audio
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)
        )
    elif emo_control_method == 2:  # emotion vectors
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    elif emo_control_method == 3:  # emotion text description
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    else:  # 0: same as speaker voice
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def on_experimental_change(is_exp):
    # 切换情感控制选项
    if is_exp:
        return gr.update(choices=EMO_CHOICES_EXPERIMENTAL, value=EMO_CHOICES_EXPERIMENTAL[0]), gr.update(visible=True), gr.update(value=example_cases)
    else:
        return gr.update(choices=EMO_CHOICES_BASE, value=EMO_CHOICES_BASE[0]), gr.update(visible=False), gr.update(value=example_cases[:-2])

def on_history_select(selected_label, history_state):
    if not selected_label:
        return gr.update(value=None, visible=False)
    # find path
    for item in (history_state or []):
        if item["label"] == selected_label:
            return gr.update(value=item["path"], visible=True)
    return gr.update(value=None, visible=False)

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()

    gr.HTML('''
    <div align="center">
      <h1 style="font-size: 2.5em;"><a href="https://getgoingfast.pro">Get Going Fast</a></h1>
    </div>
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
    <p align="center">
      <a href="https://open.spotify.com/artist/0oH5qisu13DpnT7DucnV9d">Listen to Good Music</a>
    </p>
    <p align="center">
      <a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
    </p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            yt_url = gr.Textbox(label="YouTube URL", placeholder="Enter YouTube URL to download random 8s audio")
            yt_button = gr.Button("Download Random 8s Audio")
        with gr.Row():
            save_yt_clip_checkbox = gr.Checkbox(label="Save 8s clip to ./voices", value=True)
            auto_save = gr.Checkbox(label="Auto-save generations to ./outputs", value=True)

        # New components for refining
        full_audio_preview = gr.Audio(label="Full Audio Preview", visible=False)
        with gr.Row(visible=False) as refine_row:
            start_time = gr.Number(label="Start time (s)", value=0, step=0.1, precision=1)
            end_time = gr.Number(label="End time (s)", value=8.0, step=0.1, precision=1)
            segment_preview = gr.Audio(label="Segment Preview", visible=False)
            trim_button = gr.Button("Trim & Load Segment to Prompt", visible=False)
            full_path_hidden = gr.Textbox(visible=False, label="Full Audio Path")

        with gr.Row():
            os.makedirs("prompts", exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"), key="prompt_audio", sources=["upload", "microphone"], type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]

            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"), key="input_text_single", placeholder=i18n("请输入目标文本"),
                                                info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button", interactive=True)
                output_audio = gr.Audio(label=i18n("生成结果"), visible=True, key="output_audio")
                experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)

        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_BASE,
                    type="index",
                    value=EMO_CHOICES_BASE[0],
                    label=i18n("情感控制方式")
                )

            # 情感参考音频部分
            with gr.Group(visible=False) as emotion_reference_group:
                with gr.Row():
                    emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

            # 情感随机采样
            with gr.Row(visible=False) as emotion_randomize_group:
                emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

            # 情感向量控制部分
            with gr.Group(visible=False) as emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

            with gr.Group(visible=False) as emo_text_group:
                with gr.Row():
                    emo_text = gr.Textbox(
                        label=i18n("情感描述文本"),
                        placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                        value="",
                        info=i18n("例如：委屈巴巴、危险在悄悄逼近")
                    )

            with gr.Row(visible=False) as emo_weight_group:
                emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.8, step=0.01)

            with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=False) as advanced_settings_group:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                        with gr.Row():
                            do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                            temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                        with gr.Row():
                            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                            top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                            num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                        with gr.Row():
                            repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                            length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                            max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")

                    with gr.Column(scale=2):
                        gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                        with gr.Row():
                            initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                            max_text_tokens_per_segment = gr.Slider(
                                label=i18n("分句最大Token数"),
                                value=initial_value,
                                minimum=20,
                                maximum=tts.cfg.gpt.max_text_tokens,
                                step=2,
                                key="max_text_tokens_per_segment",
                                info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                            )

                        with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                            segments_preview = gr.Dataframe(
                                headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                                key="segments_preview",
                                wrap=True,
                            )

        # Session history UI
        with gr.Accordion("Session history", open=False):
            history_state = gr.State([])
            history_dropdown = gr.Dropdown(label="Generations", choices=[], interactive=True)
            history_audio = gr.Audio(label="Play selection", visible=False)

        advanced_params = [
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
        ]

        if len(example_cases) > 2:
            example_table = gr.Examples(
                examples=example_cases[:-2],
                examples_per_page=20,
                inputs=[prompt_audio, emo_control_method, input_text_single, emo_upload, emo_weight, emo_text, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, experimental_checkbox]
            )
        elif len(example_cases) > 0:
            example_table = gr.Examples(
                examples=example_cases,
                examples_per_page=20,
                inputs=[prompt_audio, emo_control_method, input_text_single, emo_upload, emo_weight, emo_text, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, experimental_checkbox]
            )

        # wiring
        emo_control_method.select(
            on_method_select,
            inputs=[emo_control_method],
            outputs=[emotion_reference_group, emotion_randomize_group, emotion_vector_group, emo_text_group, emo_weight_group]
        )
        input_text_single.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment],
            outputs=[segments_preview]
        )
        experimental_checkbox.change(
            on_experimental_change,
            inputs=[experimental_checkbox],
            outputs=[emo_control_method, advanced_settings_group, example_table.dataset]  # 高级参数Accordion
        )
        max_text_tokens_per_segment.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment],
            outputs=[segments_preview]
        )
        prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])

        gen_button.click(
            gen_single,
            inputs=[
                emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_segment,
                *advanced_params,
                auto_save, history_state
            ],
            outputs=[output_audio, history_dropdown, history_state]
        )

        yt_button.click(
            download_yt_audio,
            inputs=[yt_url, save_yt_clip_checkbox],
            outputs=[prompt_audio, full_audio_preview, start_time, end_time, segment_preview, trim_button, full_path_hidden, refine_row]
        )
        start_time.change(preview_segment, inputs=[start_time, end_time, full_path_hidden], outputs=[segment_preview])
        end_time.change(preview_segment, inputs=[start_time, end_time, full_path_hidden], outputs=[segment_preview])
        trim_button.click(trim_and_load, inputs=[start_time, end_time, full_path_hidden, save_yt_clip_checkbox], outputs=[prompt_audio])

        history_dropdown.change(on_history_select, inputs=[history_dropdown, history_state], outputs=[history_audio])

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
