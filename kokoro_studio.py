"""
Kokoro TTS Studio - A comprehensive Gradio UI for Kokoro TTS
Three tabs:
1. Single Speaker
2. Multi Speaker – Raw Text
3. SRT Dubbing
"""

import os
import re
import uuid
import wave
import json
import shutil
import numpy as np
from kokoro import KPipeline
from huggingface_hub import list_repo_files
from pydub import AudioSegment
import gradio as gr
from deep_translator import GoogleTranslator
import pysrt
import math

# ==================== Global Variables ====================
last_used_language = "a"
pipeline = KPipeline(lang_code=last_used_language)
temp_folder = "./kokoro_output"
os.makedirs(temp_folder, exist_ok=True)

# ==================== Voice Configuration ====================
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
    "American English": "a",
    "British English": "b",
    "Hindi": "h",
    "Spanish": "e",
    "French": "f",
    "Italian": "i",
    "Brazilian Portuguese": "p",
    "Japanese": "j",
    "Mandarin Chinese": "z",
}

LANGUAGE_MAP_LOCAL = {
    "American English": "en",
    "British English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Brazilian Portuguese": "pt",
    "Japanese": "ja",
    "Mandarin Chinese": "zh-CN",
}

# ==================== Utility Functions ====================

def get_voice_names(repo_id="hexgrad/Kokoro-82M"):
    """Fetches voice names from Hugging Face repository."""
    try:
        return sorted([
            os.path.splitext(file.replace("voices/", ""))[0]
            for file in list_repo_files(repo_id)
            if file.startswith("voices/")
        ])
    except Exception:
        all_voices = []
        for voices in VOICE_CATEGORIES.values():
            all_voices.extend(voices)
        return all_voices

def clean_text(text):
    """Clean text for TTS generation."""
    replacements = {"–": " ", "-": " ", "**": " ", "*": " ", "#": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|'
        r'[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|'
        r'[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]|'
        r'[\U00002702-\U000027B0]|[\U0001F1E0-\U0001F1FF]', flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_unique_filename(prefix="kokoro"):
    """Generate a unique filename for output audio."""
    random_string = uuid.uuid4().hex[:8]
    return os.path.join(temp_folder, f"{prefix}_{random_string}.wav")

# ==================== Audio Generation ====================

def generate_audio_single(text, voice, language="American English", speed=1.0):
    """Generate audio for single speaker."""
    global pipeline, last_used_language
    
    if not text.strip():
        return None
    
    text = clean_text(text)
    lang_code = LANGUAGE_MAP.get(language, "a")
    
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except Exception as e:
            gr.Warning(f"Fallback to English: {str(e)}")
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"
    
    try:
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
        audio_chunks = []
        
        for result in generator:
            audio = result.audio
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_chunks.append(audio_int16.tobytes())
        
        if not audio_chunks:
            gr.Error("No audio generated. Check if text is valid.")
            return None
        
        combined_audio = b''.join(audio_chunks)
        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
        
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
    """Generate audio for a speaker line."""
    global pipeline, last_used_language
    
    if not text.strip():
        return None, 0
    
    text = clean_text(text)
    lang_code = LANGUAGE_MAP.get(language, "a")
    
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except Exception:
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"
    
    try:
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
        audio_chunks = []
        
        for result in generator:
            audio = result.audio
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_chunks.append(audio_int16.tobytes())
        
        if not audio_chunks:
            return None, 0
        
        combined_audio = b''.join(audio_chunks)
        audio_array = np.frombuffer(combined_audio, dtype=np.int16)
        duration = len(audio_array) / 24000
        
        return audio_array, duration
    
    except Exception as e:
        gr.Error(f"Audio generation failed: {str(e)}")
        return None, 0

def create_multispeaker_audio(script_lines, speaker_voices, language="American English", 
                               speed=1.0, pause_between_lines=0.2, normalize_volumes=True):
    """Create multi-speaker audio from script lines."""
    all_audio = []
    speaker_volumes = {}
    
    for line_data in script_lines:
        speaker = line_data.get("speaker")
        text = line_data.get("text")
        pause_override = line_data.get("pause")
        
        # Skip pause directives
        if speaker == "_PAUSE_":
            continue
        
        voice = speaker_voices.get(speaker)
        if not voice:
            gr.Warning(f"No voice assigned for speaker '{speaker}', using af_bella")
            voice = "af_bella"
        
        audio_array, duration = generate_speaker_audio(text, voice, language, speed)
        
        if audio_array is not None and len(audio_array) > 0:
            if normalize_volumes:
                rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
                speaker_volumes[speaker] = speaker_volumes.get(speaker, []) + [rms]
            
            all_audio.append({"speaker": speaker, "audio": audio_array, "pause": pause_override})
    
    # Normalize volumes
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
    
    # Combine audio with pauses
    combined = []
    for i, item in enumerate(all_audio):
        combined.append(item["audio"])
        
        # Determine pause duration
        if item["pause"] is not None:
            pause_samples = int(24000 * item["pause"])
        elif i < len(all_audio) - 1:
            pause_samples = int(24000 * pause_between_lines)
        else:
            pause_samples = 0
        
        if pause_samples > 0:
            pause_audio = np.zeros(pause_samples, dtype=np.int16)
            combined.append(pause_audio)
    
    if not combined:
        gr.Error("No audio was generated. Check your script and speaker configuration.")
        return None
    
    final_audio = np.concatenate(combined)
    
    save_path = generate_unique_filename("multispeaker")
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(final_audio.tobytes())
    
    return save_path

# ==================== Single Speaker Tab ====================

def process_single_speaker(text, voice, speaker_name, language, speed, auto_translate):
    """Process single speaker TTS."""
    if not text or not text.strip():
        gr.Error("Please enter some text")
        return None, "❌ No text entered"
    
    if auto_translate:
        try:
            lang_code = LANGUAGE_MAP_LOCAL.get(language, "en")
            text = GoogleTranslator(target=lang_code).translate(text)
        except Exception:
            gr.Warning("Translation failed, using original text")
    
    output_path = generate_audio_single(text, voice, language, speed)
    
    if output_path:
        info = f"✅ Generated audio for speaker: **{speaker_name}**\n\n- Voice: `{voice}`\n- Language: {language}\n- Speed: {speed}x\n- Characters: {len(text)}"
        return output_path, info
    return None, "❌ Audio generation failed"

# ==================== Multi Speaker - Raw Text Tab ====================

def parse_raw_script(script_text):
    """Parse raw script text into speaker lines."""
    lines = []
    speakers = set()
    
    for line in script_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check for pause directive FIRST - before checking for speaker format
        pause_match = re.match(r'^\{pause:\s*([\d.]+)\}$', line)
        if pause_match:
            lines.append({"speaker": "_PAUSE_", "text": "", "pause": float(pause_match.group(1))})
            continue
        
        # Try speaker format: SpeakerName: text
        match = re.match(r'^([^:]+):\s*(.+)$', line)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            speakers.add(speaker)
            lines.append({"speaker": speaker, "text": text, "pause": None})
        else:
            # Continuation of previous speaker (skip pause lines)
            if lines and lines[-1]["speaker"] != "_PAUSE_":
                lines[-1]["text"] += " " + line
    
    return lines, list(speakers)

def process_raw_text_multispeaker(script_text, speaker_voices, language, speed, pause_duration, normalize_audio, auto_translate):
    """Process raw text multi-speaker script."""
    if not script_text or not script_text.strip():
        gr.Error("Please enter script text")
        return None, None, "No script text entered", {}
    
    # Parse script
    script_lines, detected_speakers = parse_raw_script(script_text)
    
    if not script_lines:
        gr.Error("Could not parse script. Use format: Speaker Name: dialogue")
        return None, None, "❌ Could not parse script", {}
    
    # Filter out pause-only lines for speaker detection
    actual_speakers = [s for s in detected_speakers if s != "_PAUSE_"]
    
    # Ensure all speakers have voices
    for speaker in actual_speakers:
        if speaker not in speaker_voices:
            speaker_voices[speaker] = "af_bella"
    
    # Translate if requested
    if auto_translate:
        try:
            lang_code = LANGUAGE_MAP_LOCAL.get(language, "en")
            for line in script_lines:
                if line["speaker"] != "_PAUSE_" and line["text"]:
                    line["text"] = GoogleTranslator(target=lang_code).translate(line["text"])
        except Exception:
            gr.Warning("Translation failed for some lines")
    
    # Generate audio
    output_path = create_multispeaker_audio(
        script_lines=script_lines,
        speaker_voices=speaker_voices,
        language=language,
        speed=speed,
        pause_between_lines=pause_duration,
        normalize_volumes=normalize_audio
    )
    
    # Generate info
    info_lines = [f"### 📋 Generated Audio"]
    info_lines.append(f"- **Speakers:** {len(actual_speakers)}")
    info_lines.append(f"- **Lines:** {len([l for l in script_lines if l['speaker'] != '_PAUSE_'])}")
    info_lines.append(f"- **Language:** {language}")
    info_lines.append(f"- **Speed:** {speed}x")
    info_lines.append("")
    info_lines.append("### 🎭 Speaker Voices:")
    for speaker, voice in speaker_voices.items():
        info_lines.append(f"- **{speaker}:** `{voice}`")
    
    info = "\n".join(info_lines)
    
    if output_path:
        return output_path, output_path, info, speaker_voices
    return None, None, "❌ Audio generation failed", speaker_voices

def update_raw_speaker_cards(script, current_voices):
    """Update speaker cards when script changes."""
    if not script or not script.strip():
        return '<p class="no-speakers">Enter script text to see speaker cards</p>', current_voices if current_voices else {}
    
    _, speakers = parse_raw_script(script)
    speakers = [s for s in speakers if s != "_PAUSE_"]
    
    if not speakers:
        return '<p class="no-speakers">No speakers detected. Use format: Speaker Name: dialogue</p>', current_voices if current_voices else {}
    
    # Initialize voices for new speakers
    if not current_voices:
        current_voices = {}
    for speaker in speakers:
        if speaker not in current_voices:
            current_voices[speaker] = "af_bella"
    
    # Generate cards
    cards = []
    for speaker in speakers:
        voice = current_voices.get(speaker, "af_bella")
        
        voice_opts = ""
        for cat, voices in VOICE_CATEGORIES.items():
            voice_opts += f'<optgroup label="{cat}">'
            for v in voices:
                sel = 'selected' if v == voice else ''
                voice_opts += f'<option value="{v}" {sel}>{v}</option>'
            voice_opts += '</optgroup>'
        
        cards.append(f'''
        <div class="speaker-card" data-speaker="{speaker}">
            <div class="speaker-card-header">
                <span style="color: var(--text-primary); font-weight: bold;">{speaker}</span>
            </div>
            <div class="speaker-voice-select">
                <label style="color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; display: block;">Voice:</label>
                <select class="voice-dropdown" data-speaker="{speaker}" onchange="window.rawVoice_{speaker} = this.value">
                    {voice_opts}
                </select>
            </div>
        </div>
        ''')
    
    return ''.join(cards), current_voices

def update_raw_voice(speaker, voice, current_voices):
    """Update voice for a speaker."""
    if not current_voices:
        current_voices = {}
    current_voices[speaker] = voice
    return current_voices

# ==================== SRT Dubbing Tab ====================

def parse_srt_file(srt_file_path):
    """Parse SRT file and return subtitles."""
    subs = pysrt.open(srt_file_path)
    return subs

def create_srt_audio(subs, speaker_voices, language="American English", speed=1.0, 
                     match_timing=True, translate_text=False, target_language="American English"):
    """Generate audio from SRT subtitles."""
    global pipeline, last_used_language
    
    lang_code = LANGUAGE_MAP.get(language, "a")
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except Exception:
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"
    
    output_dir = "./kokoro_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get default voice
    speaker_names = list(speaker_voices.keys())
    default_voice = speaker_voices.get(speaker_names[0], "af_bella") if speaker_names else "af_bella"
    
    # Generate audio for each subtitle
    audio_segments = []
    
    for i, sub in enumerate(subs):
        text = sub.text.replace('\n', ' ')
        
        # Translate if requested
        if translate_text:
            try:
                lang_code_target = LANGUAGE_MAP_LOCAL.get(target_language, "en")
                text = GoogleTranslator(target=lang_code_target).translate(text)
                sub.text = text
            except Exception:
                gr.Warning(f"Translation failed for subtitle {i+1}")
        
        # Calculate duration needed
        duration_needed = (sub.end.ordinal - sub.start.ordinal) / 1000.0
        
        # Generate audio
        try:
            # Try to match timing by adjusting speed
            effective_speed = speed
            if match_timing and duration_needed > 0:
                # Estimate: ~15 characters per second at normal speed
                estimated_duration = len(text) / 15.0
                if estimated_duration > duration_needed:
                    effective_speed = min(speed * (estimated_duration / duration_needed), 2.0)
            
            generator = pipeline(text, voice=default_voice, speed=effective_speed, split_pattern=r'\n+')
            audio_chunks = []
            
            for result in generator:
                audio = result.audio
                audio_np = audio.numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
            
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                audio_segments.append(audio_array)
        except Exception as e:
            gr.Warning(f"Failed to generate audio for subtitle {i+1}: {str(e)}")
    
    if not audio_segments:
        gr.Error("No audio was generated")
        return None, None
    
    # Combine all audio segments
    combined_audio = np.concatenate(audio_segments)
    
    # Save audio
    save_path = generate_unique_filename("srt_dub")
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(combined_audio.tobytes())
    
    # Save updated SRT
    srt_output_path = save_path.replace(".wav", "_dubbed.srt")
    subs.save(srt_output_path)
    
    return save_path, srt_output_path

def process_srt_dubbing(srt_file, speaker_config, language, speed, match_timing, 
                        translate_text, target_language):
    """Process SRT dubbing."""
    if not srt_file:
        gr.Error("Please upload an SRT file")
        return None, None, None, "No SRT file uploaded"
    
    # Parse speaker config
    speaker_voices = {}
    for line in speaker_config.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker_voices[parts[0].strip()] = parts[1].strip()
    
    if not speaker_voices:
        speaker_voices = {"Speaker1": "af_bella"}
    
    try:
        subs = parse_srt_file(srt_file)
        
        audio_path, srt_output_path = create_srt_audio(
            subs=subs,
            speaker_voices=speaker_voices,
            language=language,
            speed=speed,
            match_timing=match_timing,
            translate_text=translate_text,
            target_language=target_language
        )
        
        if audio_path:
            info = f"✅ SRT Dubbing Complete\n\n- **Subtitles:** {len(subs)}\n- **Language:** {language}\n- **Speed:** {speed}x\n- **Match Timing:** {match_timing}"
            return audio_path, srt_output_path, audio_path, info
        return None, None, None, "❌ Audio generation failed"
    
    except Exception as e:
        gr.Error(f"SRT processing failed: {str(e)}")
        return None, None, None, f"❌ Error: {str(e)}"

# ==================== Gradio UI ====================

def create_ui():
    """Create the main Gradio UI."""
    
    # Custom CSS
    custom_css = """
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #0f3460;
        --text-primary: #e8e8e8;
        --text-secondary: #a0a0a0;
        --accent-blue: #4285f4;
        --accent-green: #34a853;
        --accent-yellow: #fbbc04;
        --accent-red: #ea4335;
        --border-color: #2a2a4e;
    }
    
    .gradio-container {
        background: var(--bg-primary) !important;
    }
    
    .gr-tabs .tab-nav {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .gr-tabs .tab-nav button {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .gr-tabs .tab-nav button.selected {
        background: var(--accent-blue) !important;
        color: white !important;
    }
    
    .gr-box {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }
    
    .gr-input, .gr-textarea, .gr-dropdown, .gr-number {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .gr-button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .gr-button-primary {
        background: var(--accent-blue) !important;
        border: none !important;
    }
    
    .speaker-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }
    
    .speaker-card-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 12px !important;
        gap: 12px !important;
    }
    
    .speaker-name-input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        flex: 1 !important;
    }
    
    .remove-speaker-btn {
        background: var(--accent-red) !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        flex-shrink: 0 !important;
    }
    
    .voice-dropdown {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        width: 100% !important;
    }
    
    .turn-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 12px !important;
    }
    
    .speaker-dot {
        width: 12px !important;
        height: 12px !important;
        border-radius: 50% !important;
        flex-shrink: 0 !important;
        margin-top: 10px !important;
    }
    
    .turn-content {
        flex: 1 !important;
        min-width: 0 !important;
    }
    
    .turn-speaker-select {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin-bottom: 8px !important;
        font-size: 13px !important;
        width: 100% !important;
    }
    
    .turn-text-input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        padding: 10px 12px !important;
        border-radius: 6px !important;
        width: 100% !important;
        min-height: 60px !important;
        resize: vertical !important;
        font-family: inherit !important;
    }
    
    .raw-preview {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 16px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 13px !important;
        white-space: pre-wrap !important;
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    
    .no-speakers {
        color: var(--text-secondary) !important;
        font-style: italic !important;
        padding: 20px !important;
        text-align: center !important;
    }
    
    .speaker-cards-container {
        max-height: 400px;
        overflow-y: auto !important;
    }
    
    select option {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    .gr-markdown {
        color: var(--text-primary) !important;
    }
    """
    
    # Get voice list
    voice_names = get_voice_names()
    
    with gr.Blocks(css=custom_css, title="Kokoro TTS Studio", theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # 🎙️ Kokoro TTS Studio
        
        Professional text-to-speech interface with dark theme
        """)
        
        with gr.Tabs() as tabs:
            # ==================== Single Speaker Tab ====================
            with gr.TabItem("🎤 Single Speaker", id="single"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Enter Text")
                        single_text = gr.Textbox(
                            label="",
                            placeholder="Enter your text here...",
                            lines=8
                        )
                        
                        single_voice = gr.Dropdown(
                            choices=voice_names,
                            label="🎙️ Voice",
                            value="af_bella",
                            interactive=True
                        )
                        
                        single_speaker_name = gr.Textbox(
                            label="📛 Speaker Name",
                            placeholder="Enter speaker name (for reference)",
                            value="Speaker 1"
                        )
                        
                        with gr.Row():
                            single_language = gr.Dropdown(
                                choices=list(LANGUAGE_MAP.keys()),
                                label="🌍 Language",
                                value="American English"
                            )
                            single_speed = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="⚡ Speed"
                            )
                        
                        single_translate = gr.Checkbox(
                            label="🌐 Auto-translate to target language",
                            value=False
                        )
                        
                        single_generate = gr.Button(
                            "🚀 Generate Audio",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎧 Output")
                        single_audio = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                            autoplay=True
                        )
                        single_audio_file = gr.File(label="📥 Download Audio")
                        single_info = gr.Markdown("Click Generate to create audio...")
                
                gr.Examples(
                    examples=[
                        ["Hello! Welcome to Kokoro TTS Studio.", "af_bella", "Narrator"],
                        ["Hey there! How's it going? I hope you're having a wonderful day.", "am_adam", "Host"],
                        ["This is a demonstration of the single speaker text to speech functionality.", "bf_isabella", "Assistant"],
                        ["The quick brown fox jumps over the lazy dog.", "af_nicole", "Speaker"],
                        ["In a world full of noise, be the silence that speaks volumes.", "bm_george", "Narrator"],
                    ],
                    inputs=[single_text, single_voice, single_speaker_name]
                )
                
                # Connect generate button
                single_generate.click(
                    fn=process_single_speaker,
                    inputs=[single_text, single_voice, single_speaker_name, single_language, 
                           single_speed, single_translate],
                    outputs=[single_audio, single_info]
                )
            
            # ==================== Multi Speaker - Raw Text Tab ====================
            with gr.TabItem("👥 Multi Speaker – Raw Text", id="multi_raw"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Script (Paste your text here)")
                        gr.Markdown("**Format:** `Speaker Name: dialogue text`")
                        raw_script = gr.Textbox(
                            label="",
                            placeholder="""Speaker 1: Hello there!
Speaker 2: Hi, how are you?
Speaker 1: I'm doing great, thanks for asking!
Speaker 2: That's wonderful to hear.
{pause: 0.5}
Speaker 1: Would you like to grab some coffee?
Speaker 2: That sounds wonderful!""",
                            lines=15
                        )
                        
                        gr.Markdown("**Examples:**")
                        gr.Markdown("- `Speaker1: Hello!`")
                        gr.Markdown("- `{pause: 0.5}` for custom pause")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎭 Speaker Voices")
                        
                        # Hidden state for speaker voices
                        raw_voice_state = gr.JSON(value={})
                        
                        raw_speaker_cards = gr.HTML(value='<p class="no-speakers">Enter script text to see speaker cards</p>')
                
                with gr.Row():
                    raw_language = gr.Dropdown(
                        choices=list(LANGUAGE_MAP.keys()),
                        label="🌍 Language",
                        value="American English"
                    )
                    raw_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="⚡ Speed"
                    )
                    raw_pause = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.2,
                        step=0.1,
                        label="⏸️ Pause Between Lines (seconds)"
                    )
                
                with gr.Row():
                    raw_normalize = gr.Checkbox(label="🔊 Normalize Speaker Volumes", value=True)
                    raw_translate = gr.Checkbox(label="🌐 Auto-translate to target language", value=False)
                
                raw_generate = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
                
                gr.Markdown("### 🎧 Output")
                raw_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)
                raw_audio_file = gr.File(label="📥 Download Audio")
                raw_info = gr.Markdown("Click Generate to create multi-speaker audio...")
                
                # Update speaker cards on script change
                raw_script.change(
                    fn=update_raw_speaker_cards,
                    inputs=[raw_script, raw_voice_state],
                    outputs=[raw_speaker_cards, raw_voice_state]
                )
                
                # Generate audio - pass voice_state directly
                raw_generate.click(
                    fn=process_raw_text_multispeaker,
                    inputs=[raw_script, raw_voice_state, raw_language, raw_speed, 
                           raw_pause, raw_normalize, raw_translate],
                    outputs=[raw_audio, raw_audio_file, raw_info, raw_voice_state]
                )
                
                gr.Examples(
                    examples=[
                        ["""Speaker 1: Hello! Welcome to our podcast.
Speaker 2: Thanks for having me!
Speaker 1: Today we're discussing AI technology.
{pause: 0.5}
Speaker 2: Exciting topic! Let's dive in."""],
                        ["""John: Hey, how's it going?
Mary: Pretty good! Just finished a big project.
John: That's awesome! We should celebrate.
Mary: Definitely! Coffee tomorrow?
John: Sounds perfect!"""],
                        ["""Narrator: It was a dark and stormy night.
Hero: I must find the treasure!
Villain: Not if I find it first!
{pause: 1.0}
Narrator: The battle was about to begin."""],
                    ],
                    inputs=[raw_script]
                )
            
            # ==================== SRT Dubbing Tab ====================
            with gr.TabItem("🎬 SRT Dubbing", id="srt_dub"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📹 Upload SRT File")
                        srt_input = gr.File(
                            label="Upload SRT Subtitle File",
                            file_types=[".srt"]
                        )
                        
                        gr.Markdown("### 🎭 Speaker Configuration")
                        gr.Markdown("Format: `SpeakerName:voice_name`")
                        srt_speaker_config = gr.Textbox(
                            label="Speaker Config",
                            value="Speaker1:af_bella",
                            lines=4,
                            placeholder="Speaker1:af_bella\nSpeaker2:bf_isabella"
                        )
                        
                        with gr.Row():
                            srt_language = gr.Dropdown(
                                choices=list(LANGUAGE_MAP.keys()),
                                label="🌍 Language",
                                value="American English"
                            )
                            srt_speed = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="⚡ Speed"
                            )
                        
                        srt_match_timing = gr.Checkbox(
                            label="⏱️ Match Audio to Subtitle Timing",
                            value=True,
                            info="Adjust speed to fit subtitle duration"
                        )
                        
                        srt_translate = gr.Checkbox(
                            label="🌐 Translate Subtitles",
                            value=False
                        )
                        
                        srt_target_language = gr.Dropdown(
                            choices=list(LANGUAGE_MAP.keys()),
                            label="🎯 Target Language",
                            value="American English",
                            visible=False
                        )
                        
                        srt_translate.change(
                            lambda x: gr.update(visible=x),
                            inputs=[srt_translate],
                            outputs=[srt_target_language]
                        )
                        
                        srt_generate = gr.Button("🚀 Generate Dubbed Audio", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎧 Output")
                        srt_audio = gr.Audio(
                            label="Dubbed Audio",
                            type="filepath",
                            autoplay=True
                        )
                        srt_audio_file = gr.File(label="📥 Download Audio")
                        srt_srt_file = gr.File(label="📜 Download Updated SRT")
                        srt_info = gr.Markdown("Upload SRT and click Generate...")
                
                srt_generate.click(
                    fn=process_srt_dubbing,
                    inputs=[srt_input, srt_speaker_config, srt_language, srt_speed,
                           srt_match_timing, srt_translate, srt_target_language],
                    outputs=[srt_audio, srt_srt_file, srt_audio_file, srt_info]
                )
                
                gr.Examples(
                    examples=[
                        ["Speaker1:af_bella", "American English", 1.0, True],
                        ["Speaker1:bf_isabella\nSpeaker2:bm_george", "British English", 1.1, True],
                    ],
                    inputs=[srt_speaker_config, srt_language, srt_speed, srt_match_timing]
                )
        
        gr.Markdown("""
        ---
        ### 💡 Tips
        - **Single Speaker:** Quick TTS with any voice
        - **Raw Text:** Paste dialogue in `Speaker: text` format, use `{pause: 0.5}` for pauses
        - **SRT Dubbing:** Generate audio from subtitle files
        
        **Voice Categories:**
        - American/British English: Most voices available
        - Other languages: Hindi, Spanish, French, Italian, Portuguese, Japanese, Chinese
        """)
    
    return demo

# ==================== Entry Point ====================

import click

@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable public sharing.")
@click.option("--port", type=int, default=7860, help="Port to run on.")
def main(debug, share, port):
    demo = create_ui()
    demo.queue().launch(debug=debug, share=share, server_port=port)
    print(f"\n🎙️ Kokoro TTS Studio running at: http://localhost:{port}")
    if share:
        print("🌍 Public URL will be shown above when available\n")

if __name__ == "__main__":
    main()
