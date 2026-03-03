"""
Kokoro TTS Studio - Clean Gradio UI for Kokoro TTS
Three tabs: Single Speaker, Multi Speaker – Raw Text, SRT Dubbing
"""

import os
import re
import uuid
import wave
import json
import numpy as np
from kokoro import KPipeline
from huggingface_hub import list_repo_files
import gradio as gr
from deep_translator import GoogleTranslator
import pysrt

# ==================== Globals ====================
last_used_language = "a"
pipeline = KPipeline(lang_code=last_used_language)
temp_folder = "./kokoro_output"
os.makedirs(temp_folder, exist_ok=True)

VOICE_CATEGORIES = {
    "American Female": ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_sky", "af_sarah", "af_nova", "af_river"],
    "American Male": ["am_adam", "am_michael", "am_echo", "am_eric", "am_liam", "am_onyx"],
    "British Female": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"],
    "British Male": ["bm_george", "bm_lewis", "bm_daniel", "bm_fable"],
    "Hindi": ["hf_alpha", "hf_beta"],
    "Spanish": ["ef_dora", "em_alex"],
    "French": ["ff_siwis", "fm_remy"],
    "Italian": ["if_sara", "im_marco"],
    "Brazilian Portuguese": ["pf_dora", "pm_rafael"],
    "Japanese": ["jf_nezumi", "jm_kumo"],
    "Mandarin Chinese": ["zf_xiaoni", "zm_yunjian"],
}

LANGUAGE_MAP = {
    "American English": "a", "British English": "b", "Hindi": "h", "Spanish": "e",
    "French": "f", "Italian": "i", "Brazilian Portuguese": "p", "Japanese": "j", "Mandarin Chinese": "z",
}

LANGUAGE_MAP_LOCAL = {
    "American English": "en", "British English": "en", "Hindi": "hi", "Spanish": "es",
    "French": "fr", "Italian": "it", "Brazilian Portuguese": "pt", "Japanese": "ja", "Mandarin Chinese": "zh-CN",
}

# ==================== Utilities ====================

def get_voice_names(repo_id="hexgrad/Kokoro-82M"):
    try:
        return sorted([os.path.splitext(f.replace("voices/", ""))[0] 
                      for f in list_repo_files(repo_id) if f.startswith("voices/")])
    except:
        return [v for voices in VOICE_CATEGORIES.values() for v in voices]

def clean_text(text):
    replacements = {"–": " ", "-": " ", "**": " ", "*": " ", "#": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|'
        r'[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|'
        r'[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]|'
        r'[\U00002702-\U000027B0]|[\U0001F1E0-\U0001F1FF]', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return re.sub(r'\s+', ' ', text).strip()

def generate_unique_filename(prefix="kokoro"):
    return os.path.join(temp_folder, f"{prefix}_{uuid.uuid4().hex[:8]}.wav")

# ==================== Audio Generation ====================

def generate_audio_single(text, voice, language="American English", speed=1.0):
    global pipeline, last_used_language
    if not text.strip():
        return None
    text = clean_text(text)
    lang_code = LANGUAGE_MAP.get(language, "a")
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except:
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"
    try:
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
        audio_chunks = []
        for result in generator:
            audio_np = result.audio.numpy()
            audio_chunks.append((audio_np * 32767).astype(np.int16).tobytes())
        if not audio_chunks:
            return None
        audio_array = np.frombuffer(b''.join(audio_chunks), dtype=np.int16)
        save_path = generate_unique_filename("single")
        with wave.open(save_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_array.tobytes())
        return save_path
    except Exception as e:
        gr.Error(f"Audio generation failed: {str(e)}")
        return None

def generate_speaker_audio(text, voice, language="American English", speed=1.0):
    global pipeline, last_used_language
    if not text.strip():
        return None, 0
    text = clean_text(text)
    lang_code = LANGUAGE_MAP.get(language, "a")
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except:
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"
    try:
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
        audio_chunks = []
        for result in generator:
            audio_np = result.audio.numpy()
            audio_chunks.append((audio_np * 32767).astype(np.int16).tobytes())
        if not audio_chunks:
            return None, 0
        audio_array = np.frombuffer(b''.join(audio_chunks), dtype=np.int16)
        return audio_array, len(audio_array) / 24000
    except Exception as e:
        gr.Error(f"Audio generation failed: {str(e)}")
        return None, 0

def create_multispeaker_audio(script_lines, speaker_voices, language="American English", 
                               speed=1.0, pause_between_lines=0.2, normalize_volumes=True):
    all_audio = []
    speaker_volumes = {}
    for line_data in script_lines:
        speaker = line_data.get("speaker")
        text = line_data.get("text")
        pause_override = line_data.get("pause")
        if speaker == "_PAUSE_":
            continue
        voice = speaker_voices.get(speaker)
        if not voice:
            voice = "af_bella"
        audio_array, duration = generate_speaker_audio(text, voice, language, speed)
        if audio_array is not None and len(audio_array) > 0:
            if normalize_volumes:
                rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
                speaker_volumes[speaker] = speaker_volumes.get(speaker, []) + [rms]
            all_audio.append({"speaker": speaker, "audio": audio_array, "pause": pause_override})
    if normalize_volumes and speaker_volumes:
        avg_volumes = {spk: np.mean(vols) for spk, vols in speaker_volumes.items()}
        global_avg = np.mean(list(avg_volumes.values())) if avg_volumes else 1.0
        normalized_audio = []
        for item in all_audio:
            speaker = item["speaker"]
            audio_array = item["audio"]
            if avg_volumes.get(speaker, 0) > 0:
                scale_factor = min(global_avg / avg_volumes[speaker], 2.0)
                audio_normalized = (audio_array.astype(np.float32) * scale_factor).astype(np.int16)
                normalized_audio.append({"speaker": speaker, "audio": audio_normalized, "pause": item["pause"]})
            else:
                normalized_audio.append(item)
        all_audio = normalized_audio
    combined = []
    for i, item in enumerate(all_audio):
        combined.append(item["audio"])
        if item["pause"] is not None:
            pause_samples = int(24000 * item["pause"])
        elif i < len(all_audio) - 1:
            pause_samples = int(24000 * pause_between_lines)
        else:
            pause_samples = 0
        if pause_samples > 0:
            combined.append(np.zeros(pause_samples, dtype=np.int16))
    if not combined:
        gr.Error("No audio was generated.")
        return None
    final_audio = np.concatenate(combined)
    save_path = generate_unique_filename("multispeaker")
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(final_audio.tobytes())
    return save_path

# ==================== Single Speaker ====================

def process_single_speaker(text, voice, speaker_name, language, speed, auto_translate):
    if not text or not text.strip():
        gr.Error("Please enter some text")
        return None, "❌ No text entered"
    if auto_translate:
        try:
            lang_code = LANGUAGE_MAP_LOCAL.get(language, "en")
            text = GoogleTranslator(target=lang_code).translate(text)
        except:
            gr.Warning("Translation failed")
    output_path = generate_audio_single(text, voice, language, speed)
    if output_path:
        info = f"✅ Generated\n- Voice: `{voice}`\n- Language: {language}\n- Speed: {speed}x"
        return output_path, info
    return None, "❌ Generation failed"

# ==================== Multi Speaker - Raw Text ====================

def parse_raw_script(script_text):
    lines = []
    speakers = set()
    for line in script_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        pause_match = re.match(r'^\{pause:\s*([\d.]+)\}$', line)
        if pause_match:
            lines.append({"speaker": "_PAUSE_", "text": "", "pause": float(pause_match.group(1))})
            continue
        match = re.match(r'^([^:]+):\s*(.+)$', line)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            speakers.add(speaker)
            lines.append({"speaker": speaker, "text": text, "pause": None})
        elif lines and lines[-1]["speaker"] != "_PAUSE_":
            lines[-1]["text"] += " " + line
    return lines, [s for s in speakers if s != "_PAUSE_"]

def process_raw_text_multispeaker(script_text, speaker_config_text, language, speed, pause_duration, normalize_audio, auto_translate):
    if not script_text or not script_text.strip():
        gr.Error("Please enter script text")
        return None, None, "No script text", {}
    speaker_voices = {}
    for line in speaker_config_text.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker_voices[parts[0].strip()] = parts[1].strip()
    script_lines, detected_speakers = parse_raw_script(script_text)
    if not script_lines:
        gr.Error("Could not parse script")
        return None, None, "❌ Parse error", {}
    for speaker in detected_speakers:
        if speaker not in speaker_voices:
            speaker_voices[speaker] = "af_bella"
    if auto_translate:
        try:
            lang_code = LANGUAGE_MAP_LOCAL.get(language, "en")
            for line in script_lines:
                if line["speaker"] != "_PAUSE_" and line["text"]:
                    line["text"] = GoogleTranslator(target=lang_code).translate(line["text"])
        except:
            gr.Warning("Translation failed")
    output_path = create_multispeaker_audio(script_lines, speaker_voices, language, speed, pause_duration, normalize_audio)
    info_lines = [f"### 📋 Generated", f"- **Speakers:** {len(detected_speakers)}", f"- **Lines:** {len([l for l in script_lines if l['speaker'] != '_PAUSE_'])}"]
    info_lines.append(f"### 🎭 Voices:")
    for speaker, voice in speaker_voices.items():
        info_lines.append(f"- **{speaker}:** `{voice}`")
    if output_path:
        return output_path, output_path, "\n".join(info_lines), speaker_voices
    return None, None, "❌ Generation failed", speaker_voices

def update_speaker_config(script, current_config):
    _, speakers = parse_raw_script(script)
    voices = {}
    for line in current_config.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                voices[parts[0].strip()] = parts[1].strip()
    for speaker in speakers:
        if speaker not in voices:
            voices[speaker] = "af_bella"
    return '\n'.join([f"{s}:{v}" for s, v in voices.items()])

# ==================== SRT Dubbing ====================

def process_srt_dubbing(srt_file, speaker_config, language, speed, match_timing, translate_text, target_language):
    if not srt_file:
        gr.Error("Please upload an SRT file")
        return None, None, None, "No SRT file"
    speaker_voices = {}
    for line in speaker_config.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker_voices[parts[0].strip()] = parts[1].strip()
    if not speaker_voices:
        speaker_voices = {"Speaker1": "af_bella"}
    try:
        subs = pysrt.open(srt_file)
        default_voice = list(speaker_voices.values())[0]
        lang_code = LANGUAGE_MAP.get(language, "a")
        global pipeline, last_used_language
        if lang_code != last_used_language:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        audio_segments = []
        for i, sub in enumerate(subs):
            text = sub.text.replace('\n', ' ')
            if translate_text:
                try:
                    lang_code_target = LANGUAGE_MAP_LOCAL.get(target_language, "en")
                    text = GoogleTranslator(target=lang_code_target).translate(text)
                except:
                    pass
            try:
                effective_speed = speed
                if match_timing:
                    duration_needed = (sub.end.ordinal - sub.start.ordinal) / 1000.0
                    estimated_duration = len(text) / 15.0
                    if estimated_duration > duration_needed:
                        effective_speed = min(speed * (estimated_duration / duration_needed), 2.0)
                generator = pipeline(text, voice=default_voice, speed=effective_speed, split_pattern=r'\n+')
                for result in generator:
                    audio_np = result.audio.numpy()
                    audio_segments.append((audio_np * 32767).astype(np.int16))
            except Exception as e:
                gr.Warning(f"Subtitle {i+1} failed: {e}")
        if not audio_segments:
            return None, None, None, "❌ No audio generated"
        combined = np.concatenate(audio_segments)
        save_path = generate_unique_filename("srt_dub")
        with wave.open(save_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(combined.tobytes())
        srt_output = save_path.replace(".wav", "_dubbed.srt")
        subs.save(srt_output)
        info = f"✅ Dubbed {len(subs)} subtitles"
        return save_path, srt_output, save_path, info
    except Exception as e:
        gr.Error(f"SRT error: {str(e)}")
        return None, None, None, f"❌ Error: {str(e)}"

# ==================== UI ====================

custom_css = """
:root {
    --bg-primary: #1a1a2e; --bg-secondary: #16213e; --bg-card: #0f3460;
    --text-primary: #e8e8e8; --text-secondary: #a0a0a0;
    --accent-blue: #4285f4; --accent-red: #ea4335; --border-color: #2a2a4e;
}
.gradio-container { background: var(--bg-primary) !important; }
.gr-tabs .tab-nav { background: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; }
.gr-tabs .tab-nav button { background: var(--bg-card) !important; color: var(--text-primary) !important; }
.gr-tabs .tab-nav button.selected { background: var(--accent-blue) !important; color: white !important; }
.gr-box { background: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; border-radius: 12px !important; }
.gr-input, .gr-textarea, .gr-dropdown { background: var(--bg-card) !important; border: 1px solid var(--border-color) !important; color: var(--text-primary) !important; }
.gr-button-primary { background: var(--accent-blue) !important; }
.gr-markdown { color: var(--text-primary) !important; }
select option { background: var(--bg-card) !important; color: var(--text-primary) !important; }
"""

def create_ui():
    voice_names = get_voice_names()
    
    with gr.Blocks(css=custom_css, title="Kokoro TTS Studio", theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🎙️ Kokoro TTS Studio\n\nProfessional text-to-speech with dark theme")
        
        with gr.Tabs():
            # ========== Single Speaker ==========
            with gr.TabItem("🎤 Single Speaker"):
                with gr.Row():
                    with gr.Column(scale=2):
                        single_text = gr.Textbox(label="📝 Text", placeholder="Enter text...", lines=6)
                        single_voice = gr.Dropdown(choices=voice_names, label="🎙️ Voice", value="af_bella")
                        single_speaker_name = gr.Textbox(label="📛 Speaker Name", value="Speaker 1")
                        with gr.Row():
                            single_language = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), label="🌍 Language", value="American English")
                            single_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="⚡ Speed")
                        single_translate = gr.Checkbox(label="🌐 Auto-translate", value=False)
                        single_generate = gr.Button("🚀 Generate", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        single_audio = gr.Audio(label="🎧 Audio", type="filepath", autoplay=True)
                        single_audio_file = gr.File(label="📥 Download")
                        single_info = gr.Markdown("Click Generate...")
                
                single_generate.click(fn=process_single_speaker,
                    inputs=[single_text, single_voice, single_speaker_name, single_language, single_speed, single_translate],
                    outputs=[single_audio, single_info])
                
                gr.Examples(examples=[
                    ["Hello! Welcome to Kokoro TTS Studio.", "af_bella", "Narrator"],
                    ["Hey there! How's it going?", "am_adam", "Host"],
                    ["This is a demonstration.", "bf_isabella", "Assistant"],
                ], inputs=[single_text, single_voice, single_speaker_name])
            
            # ========== Multi Speaker - Raw Text ==========
            with gr.TabItem("👥 Multi Speaker – Raw Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Script")
                        gr.Markdown("**Format:** `Speaker: text` | `{pause: 0.5}`")
                        raw_script = gr.Textbox(label="", placeholder="""Speaker 1: Hello there!
Speaker 2: Hi, how are you?
Speaker 1: I'm doing great!
{pause: 0.5}
Speaker 2: That's wonderful!""", lines=12)
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎭 Speaker Voices")
                        gr.Markdown("Format: `Speaker:voice`")
                        speaker_config = gr.Textbox(label="Speaker Config", 
                            value="Speaker 1:af_bella\nSpeaker 2:bf_isabella", lines=6)
                
                with gr.Row():
                    raw_language = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), label="🌍 Language", value="American English")
                    raw_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="⚡ Speed")
                    raw_pause = gr.Slider(minimum=0.0, maximum=2.0, value=0.2, step=0.1, label="⏸️ Pause (s)")
                
                with gr.Row():
                    raw_normalize = gr.Checkbox(label="🔊 Normalize", value=True)
                    raw_translate = gr.Checkbox(label="🌐 Translate", value=False)
                
                raw_generate = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
                
                raw_audio = gr.Audio(label="🎧 Audio", type="filepath", autoplay=True)
                raw_audio_file = gr.File(label="📥 Download")
                raw_info = gr.Markdown("Click Generate...")
                
                raw_script.change(fn=update_speaker_config, inputs=[raw_script, speaker_config], outputs=[speaker_config])
                raw_generate.click(fn=process_raw_text_multispeaker,
                    inputs=[raw_script, speaker_config, raw_language, raw_speed, raw_pause, raw_normalize, raw_translate],
                    outputs=[raw_audio, raw_audio_file, raw_info, speaker_config])
                
                gr.Examples(examples=[
                    ["""Speaker 1: Welcome to our podcast!
Speaker 2: Thanks for having me!
{pause: 0.5}
Speaker 1: Let's discuss AI today."""],
                    ["""John: Hey, how's it going?
Mary: Pretty good! Just finished a project.
John: Awesome! Let's celebrate."""],
                ], inputs=[raw_script])
            
            # ========== SRT Dubbing ==========
            with gr.TabItem("🎬 SRT Dubbing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        srt_input = gr.File(label="📹 Upload SRT", file_types=[".srt"])
                        srt_speaker_config = gr.Textbox(label="🎭 Speaker Config", value="Speaker1:af_bella", lines=3)
                        with gr.Row():
                            srt_language = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), label="🌍 Language", value="American English")
                            srt_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="⚡ Speed")
                        srt_match_timing = gr.Checkbox(label="⏱️ Match Timing", value=True)
                        srt_translate = gr.Checkbox(label="🌐 Translate", value=False)
                        srt_target_language = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), label="🎯 Target", value="American English", visible=False)
                        srt_translate.change(lambda x: gr.update(visible=x), inputs=[srt_translate], outputs=[srt_target_language])
                        srt_generate = gr.Button("🚀 Generate Dub", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        srt_audio = gr.Audio(label="🎧 Dubbed Audio", type="filepath", autoplay=True)
                        srt_audio_file = gr.File(label="📥 Download Audio")
                        srt_srt_file = gr.File(label="📜 Download SRT")
                        srt_info = gr.Markdown("Upload SRT and click Generate...")
                
                srt_generate.click(fn=process_srt_dubbing,
                    inputs=[srt_input, srt_speaker_config, srt_language, srt_speed, srt_match_timing, srt_translate, srt_target_language],
                    outputs=[srt_audio, srt_srt_file, srt_audio_file, srt_info])
        
        gr.Markdown("\n---\n### 💡 Tips\n- **Single Speaker:** Quick TTS\n- **Raw Text:** `Speaker: text` format, `{pause: 0.5}` for pauses\n- **SRT Dubbing:** Generate audio from subtitles")
    
    return demo

import click
@click.command()
@click.option("--debug", is_flag=True, default=False)
@click.option("--share", is_flag=True, default=False)
@click.option("--port", type=int, default=7860)
def main(debug, share, port):
    demo = create_ui()
    demo.queue().launch(debug=debug, share=share, server_port=port)
    print(f"\n🎙️ Running at: http://localhost:{port}")

if __name__ == "__main__":
    main()
