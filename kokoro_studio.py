"""
Kokoro TTS Studio - A comprehensive Gradio UI for Kokoro TTS
Four tabs:
1. Single Speaker
2. Multi Speaker – Raw Text
3. Multi Speaker – Script Editor
4. SRT Dubbing
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
import librosa
import soundfile as sf
from tqdm.auto import tqdm
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
    
    text = clean_text(text)
    lang_code = LANGUAGE_MAP.get(language, "a")
    
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except Exception:
            gr.Warning(f"Fallback to English")
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
        
        voice = speaker_voices.get(speaker)
        if not voice:
            gr.Warning(f"No voice assigned for speaker '{speaker}', skipping...")
            continue
        
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
    
    # Combine audio
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
    if not text.strip():
        gr.Error("Please enter some text")
        return None, None
    
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
        
        # Check for pause directive
        pause_match = re.match(r'^\{pause:\s*([\d.]+)\}', line)
        if pause_match:
            lines.append({"speaker": "_PAUSE_", "text": line, "pause": float(pause_match.group(1))})
            continue
        
        # Try different formats
        match = re.match(r'^([^:]+):\s*(.+)$', line)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            speakers.add(speaker)
            lines.append({"speaker": speaker, "text": text, "pause": None})
        else:
            # Continuation of previous speaker
            if lines and lines[-1]["speaker"] != "_PAUSE_":
                lines[-1]["text"] += " " + line
    
    return lines, list(speakers)

def process_raw_text_multispeaker(script_text, speaker_voices, language, speed, pause_duration, normalize_audio, auto_translate):
    """Process raw text multi-speaker script."""
    if not script_text.strip():
        gr.Error("Please enter script text")
        return None, None, "No script text entered"
    
    # Parse script
    script_lines, detected_speakers = parse_raw_script(script_text)
    
    if not script_lines:
        gr.Error("Could not parse script. Use format: Speaker Name: dialogue")
        return None, None, "❌ Could not parse script"
    
    # Ensure all speakers have voices
    for speaker in detected_speakers:
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
    info_lines.append(f"- **Speakers:** {len(detected_speakers)}")
    info_lines.append(f"- **Lines:** {len(script_lines)}")
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

# ==================== Multi Speaker - Script Editor Tab ====================

def add_turn(turns_json, speakers_json, speaker_voices_json):
    """Add a new turn to the script editor."""
    turns = json.loads(turns_json) if turns_json else []
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    # Find next speaker number
    speaker_num = 1
    while f"Speaker {speaker_num}" in speakers:
        speaker_num += 1
    new_speaker = f"Speaker {speaker_num}"
    
    speakers[new_speaker] = True
    if new_speaker not in speaker_voices:
        speaker_voices[new_speaker] = "af_bella"
    
    new_turn = {
        "speaker": new_speaker,
        "text": "Enter dialogue text here...",
        "pause": None
    }
    turns.append(new_turn)
    
    # Generate turn cards HTML
    turn_cards = generate_turn_cards(turns, speakers, speaker_voices)
    
    return (turn_cards, json.dumps(turns), json.dumps(speakers), json.dumps(speaker_voices),
            generate_raw_preview(turns))

def remove_turn(turns_json, speakers_json, speaker_voices_json, turn_index):
    """Remove a turn from the script editor."""
    turns = json.loads(turns_json) if turns_json else []
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    if 0 <= turn_index < len(turns):
        removed_speaker = turns[turn_index]["speaker"]
        turns.pop(turn_index)
        
        # Remove speaker if no longer used
        used_speakers = set(t["speaker"] for t in turns)
        if removed_speaker not in used_speakers and removed_speaker.startswith("Speaker"):
            speakers.pop(removed_speaker, None)
            speaker_voices.pop(removed_speaker, None)
    
    turn_cards = generate_turn_cards(turns, speakers, speaker_voices)
    return (turn_cards, json.dumps(turns), json.dumps(speakers), json.dumps(speaker_voices),
            generate_raw_preview(turns))

def clear_all_turns():
    """Clear all turns."""
    return ('<p class="no-speakers">Click "Add Turn" to start building your script</p>', 
            '[]', '{}', '{}', '')

def update_turn(turns_json, speakers_json, speaker_voices_json, turn_index, field, value):
    """Update a turn's field."""
    turns = json.loads(turns_json) if turns_json else []
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    if 0 <= turn_index < len(turns):
        if field == "speaker":
            turns[turn_index]["speaker"] = value
        elif field == "text":
            turns[turn_index]["text"] = value
        elif field == "pause":
            turns[turn_index]["pause"] = float(value) if value else None
    
    return json.dumps(turns), generate_raw_preview(turns)

def add_speaker(speakers_json, speaker_voices_json, speaker_name):
    """Add a new speaker."""
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    if not speaker_name or not speaker_name.strip():
        gr.Warning("Please enter a speaker name")
        return json.dumps(speakers), json.dumps(speaker_voices), ""
    
    speaker_name = speaker_name.strip()
    if speaker_name in speakers:
        gr.Warning("Speaker already exists")
        return json.dumps(speakers), json.dumps(speaker_voices), ""
    
    speakers[speaker_name] = True
    speaker_voices[speaker_name] = "af_bella"
    
    return json.dumps(speakers), json.dumps(speaker_voices), ""

def remove_speaker(speakers_json, speaker_voices_json, turns_json, speaker_name):
    """Remove a speaker."""
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    turns = json.loads(turns_json) if turns_json else []
    
    # Check if speaker is used
    used_speakers = set(t["speaker"] for t in turns)
    if speaker_name in used_speakers:
        gr.Warning("Cannot remove speaker that is used in turns. Remove or reassign the turns first.")
        return json.dumps(speakers), json.dumps(speaker_voices)
    
    speakers.pop(speaker_name, None)
    speaker_voices.pop(speaker_name, None)
    
    return json.dumps(speakers), json.dumps(speaker_voices)

def update_speaker_voice(speaker_voices_json, speaker_name, voice):
    """Update a speaker's voice."""
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    speaker_voices[speaker_name] = voice
    return json.dumps(speaker_voices)

def generate_turn_cards(turns, speakers, speaker_voices):
    """Generate HTML for turn cards."""
    if not turns:
        return '<p class="no-speakers">Click "Add Turn" to start building your script</p>'
    
    colors = ["#4285f4", "#34a853", "#fbbc04", "#ea4335", "#9c27b0", "#00bcd4", "#ff5722"]
    speaker_list = list(speakers.keys())
    
    cards = []
    for i, turn in enumerate(turns):
        speaker = turn["speaker"]
        text = turn["text"]
        pause = turn.get("pause")
        
        speaker_idx = speaker_list.index(speaker) if speaker in speaker_list else 0
        color = colors[speaker_idx % len(colors)]
        
        speaker_options = "".join([f'<option value="{s}" {"selected" if s == speaker else ""}>{s}</option>' 
                                  for s in speakers.keys()])
        
        pause_html = ""
        if pause is not None:
            pause_html = f'''
            <div style="margin-top: 8px;">
                <label style="color: var(--text-secondary); font-size: 12px;">Pause: </label>
                <input type="number" class="gr-input" step="0.1" min="0" value="{pause}" 
                       onchange="updateTurnPause({i}, this.value)" style="width: 80px; display: inline-block;">
                <button onclick="removeTurnPause({i})" style="margin-left: 8px; background: var(--accent-red); color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer;">✕</button>
            </div>
            '''
        else:
            pause_html = f'''
            <button onclick="addTurnPause({i})" style="margin-top: 8px; background: var(--bg-secondary); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px;">+ Add Pause</button>
            '''
        
        card = f'''
        <div class="turn-card" data-turn="{i}">
            <div class="speaker-dot" style="background: {color};"></div>
            <div class="turn-content">
                <select class="turn-speaker-select" data-turn="{i}" onchange="updateTurnSpeaker({i}, this.value)">
                    {speaker_options}
                </select>
                <textarea class="turn-text-input" data-turn="{i}" rows="2" oninput="updateTurnText({i}, this.value)">{text}</textarea>
                {pause_html}
            </div>
            <button class="remove-turn-btn" onclick="removeTurn({i})">🗑️</button>
        </div>
        '''
        cards.append(card)
    
    return ''.join(cards)

def generate_speaker_cards(speakers, speaker_voices, turns_json):
    """Generate HTML for speaker cards."""
    speakers_dict = json.loads(speakers_json) if speakers_json else {}
    speaker_voices_dict = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    if not speakers_dict:
        return '<p class="no-speakers">Add turns to see speaker cards</p>'
    
    cards = []
    for speaker in speakers_dict.keys():
        current_voice = speaker_voices_dict.get(speaker, "af_bella")
        
        voice_options = ""
        for category, voices in VOICE_CATEGORIES.items():
            voice_options += f'<optgroup label="{category}">'
            for voice in voices:
                selected = 'selected' if voice == current_voice else ''
                voice_options += f'<option value="{voice}" {selected}>{voice}</option>'
            voice_options += '</optgroup>'
        
        card = f'''
        <div class="speaker-card" data-speaker="{speaker}">
            <div class="speaker-card-header">
                <input type="text" class="speaker-name-input" value="{speaker}" 
                       placeholder="Speaker Name" onchange="renameSpeaker('{speaker}', this.value)">
                <button class="remove-speaker-btn" onclick="removeSpeaker('{speaker}')">🗑️ Remove</button>
            </div>
            <div class="speaker-voice-select">
                <label style="color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; display: block;">Voice:</label>
                <select class="voice-dropdown" data-speaker="{speaker}" onchange="updateSpeakerVoice('{speaker}', this.value)">
                    {voice_options}
                </select>
            </div>
        </div>
        '''
        cards.append(card)
    
    return ''.join(cards)

def generate_raw_preview(turns):
    """Generate raw script preview from turns."""
    if not turns:
        return ""
    
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Speaker")
        text = turn.get("text", "")
        pause = turn.get("pause")
        
        lines.append(f"{speaker}: {text}")
        if pause is not None:
            lines.append(f"{{pause: {pause}}}")
    
    return "\n".join(lines)

def process_script_editor(turns_json, speakers_json, speaker_voices_json, 
                          language, speed, pause_duration, normalize_audio, auto_translate):
    """Process script editor multi-speaker."""
    turns = json.loads(turns_json) if turns_json else []
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    if not turns:
        gr.Error("Please add at least one turn")
        return None, None, "No turns added", ""
    
    script_lines = []
    for turn in turns:
        script_lines.append({
            "speaker": turn.get("speaker", "Speaker 1"),
            "text": turn.get("text", ""),
            "pause": turn.get("pause")
        })
    
    # Translate if requested
    if auto_translate:
        try:
            lang_code = LANGUAGE_MAP_LOCAL.get(language, "en")
            for line in script_lines:
                if line["text"]:
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
    info_lines.append(f"- **Speakers:** {len(speakers)}")
    info_lines.append(f"- **Turns:** {len(script_lines)}")
    info_lines.append(f"- **Language:** {language}")
    info_lines.append(f"- **Speed:** {speed}x")
    info_lines.append("")
    info_lines.append("### 🎭 Speaker Voices:")
    for speaker, voice in speaker_voices.items():
        info_lines.append(f"- **{speaker}:** `{voice}`")
    
    info = "\n".join(info_lines)
    raw_preview = generate_raw_preview(turns)
    
    if output_path:
        return output_path, output_path, info, raw_preview
    return None, None, "❌ Audio generation failed", raw_preview

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
    
    # Generate audio for each subtitle
    audio_segments = []
    timing_data = []
    
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
        
        # Get voice for this subtitle (use first speaker or cycle through)
        speaker_names = list(speaker_voices.keys())
        voice = speaker_voices.get(speaker_names[0], "af_bella") if speaker_names else "af_bella"
        
        # Calculate speed adjustment to match timing
        duration_needed = (sub.end.ordinal - sub.start.ordinal) / 1000.0
        
        # Generate audio
        try:
            generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
            audio_chunks = []
            
            for result in generator:
                audio = result.audio
                audio_np = audio.numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
            
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                audio_array = np.frombuffer(combined_audio, dtype=np.int16)
                
                # Adjust speed if match_timing is enabled
                if match_timing:
                    current_duration = len(audio_array) / 24000.0
                    if current_duration > 0 and duration_needed > 0:
                        speed_factor = current_duration / duration_needed
                        if speed_factor > 1.5:
                            speed_factor = 1.5
                        elif speed_factor < 0.7:
                            speed_factor = 0.7
                        
                        # Regenerate with adjusted speed
                        adjusted_speed = speed * speed_factor
                        generator = pipeline(text, voice=voice, speed=adjusted_speed, split_pattern=r'\n+')
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
                timing_data.append({
                    "start": sub.start.ordinal,
                    "end": sub.end.ordinal,
                    "text": text
                })
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
    
    .turn-pause-input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        width: 100px !important;
        margin-top: 8px !important;
    }
    
    .remove-turn-btn {
        background: var(--accent-red) !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        flex-shrink: 0 !important;
        height: fit-content !important;
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
    
    .turns-container {
        max-height: 500px;
        overflow-y: auto !important;
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
    
    # JavaScript for interactivity
    js_code = """
    <script>
    function updateTurnSpeaker(index, value) {
        // Trigger Gradio update
        const turnsState = document.querySelector('.turns-state textarea');
        if (turnsState) {
            // The actual update happens via Gradio's onchange
        }
    }
    
    function updateTurnText(index, value) {
        // Live update handled by Gradio
    }
    
    function addTurnPause(index) {
        // Trigger pause addition
    }
    
    function removeTurnPause(index) {
        // Trigger pause removal
    }
    
    function removeTurn(index) {
        // Trigger turn removal via Gradio
    }
    
    function renameSpeaker(oldName, newName) {
        // Trigger speaker rename
    }
    
    function removeSpeaker(name) {
        // Trigger speaker removal
    }
    
    function updateSpeakerVoice(name, voice) {
        // Trigger voice update
    }
    </script>
    """
    
    with gr.Blocks(css=custom_css, title="Kokoro TTS Studio", theme=gr.themes.Base()) as demo:
        gr.HTML(js_code)
        
        gr.Markdown("""
        # 🎙️ Kokoro TTS Studio
        
        Professional text-to-speech interface with dark theme
        """)
        
        voice_names = get_voice_names()
        
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
                            value="af_bella"
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
                        raw_speaker_voices = gr.JSON(value={"Speaker1": "af_bella", "Speaker2": "bf_isabella"}, visible=False)
                        
                        def update_raw_cards(script, current_voices):
                            _, speakers = parse_raw_script(script)
                            if not speakers:
                                return '<p class="no-speakers">Enter script text to see speaker cards</p>', current_voices
                            
                            # Update voices
                            for spk in speakers:
                                if spk not in current_voices:
                                    current_voices[spk] = "af_bella"
                            
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
                                        <select class="voice-dropdown" data-speaker="{speaker}" onchange="updateRawVoice('{speaker}', this.value)">
                                            {voice_opts}
                                        </select>
                                    </div>
                                </div>
                                ''')
                            
                            return ''.join(cards), current_voices
                        
                        raw_speaker_cards = gr.HTML(value='<p class="no-speakers">Enter script text to see speaker cards</p>')
                        
                        # Hidden state for JS updates
                        raw_voice_state = gr.JSON(value={}, visible=False)
                
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
                    fn=update_raw_cards,
                    inputs=[raw_script, raw_voice_state],
                    outputs=[raw_speaker_cards, raw_voice_state]
                )
                
                # Generate audio
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
            
            # ==================== Multi Speaker - Script Editor Tab ====================
            with gr.TabItem("✏️ Multi Speaker – Script Editor", id="multi_editor"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Visual Script Builder")
                        
                        # Turn cards container
                        turn_cards_html = gr.HTML(
                            value='<p class="no-speakers">Click "Add Turn" to start building your script</p>',
                            elem_classes=["turns-container"]
                        )
                        
                        with gr.Row():
                            add_turn_btn = gr.Button("➕ Add Turn", size="sm")
                            clear_turns_btn = gr.Button("🗑️ Clear All", size="sm")
                        
                        # Hidden state
                        turns_state = gr.JSON(value=[], visible=False, elem_classes=["turns-state"])
                        speakers_state = gr.JSON(value={}, visible=False)
                        speaker_voices_state = gr.JSON(value={}, visible=False)
                        
                        gr.Markdown("### 📄 Raw Script Preview")
                        raw_preview = gr.Textbox(
                            label="",
                            lines=6,
                            interactive=False,
                            elem_classes=["raw-preview"]
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎭 Speaker Voices")
                        
                        speaker_cards_html = gr.HTML(
                            value='<p class="no-speakers">Add turns to see speaker cards</p>',
                            elem_classes=["speaker-cards-container"]
                        )
                        
                        with gr.Row():
                            add_speaker_btn = gr.Button("➕ Add Speaker", size="sm")
                        new_speaker_name = gr.Textbox(
                            label="New Speaker Name",
                            placeholder="Enter speaker name",
                            visible=False
                        )
                
                with gr.Row():
                    editor_language = gr.Dropdown(
                        choices=list(LANGUAGE_MAP.keys()),
                        label="🌍 Language",
                        value="American English"
                    )
                    editor_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="⚡ Speed"
                    )
                    editor_pause = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.2,
                        step=0.1,
                        label="⏸️ Pause Between Lines (seconds)"
                    )
                
                with gr.Row():
                    editor_normalize = gr.Checkbox(label="🔊 Normalize Speaker Volumes", value=True)
                    editor_translate = gr.Checkbox(label="🌐 Auto-translate to target language", value=False)
                
                editor_generate = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
                
                gr.Markdown("### 🎧 Output")
                editor_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)
                editor_audio_file = gr.File(label="📥 Download Audio")
                editor_info = gr.Markdown("Click Generate to create multi-speaker audio...")
                
                # Add turn
                add_turn_btn.click(
                    fn=add_turn,
                    inputs=[turns_state, speakers_state, speaker_voices_state],
                    outputs=[turn_cards_html, turns_state, speakers_state, speaker_voices_state, raw_preview]
                )
                
                # Clear turns
                clear_turns_btn.click(
                    fn=clear_all_turns,
                    outputs=[turn_cards_html, turns_state, speakers_state, speaker_voices_state, raw_preview]
                )
                
                # Add speaker
                add_speaker_btn.click(
                    fn=add_speaker,
                    inputs=[speakers_state, speaker_voices_state, new_speaker_name],
                    outputs=[speakers_state, speaker_voices_state, new_speaker_name]
                )
                
                # Generate
                editor_generate.click(
                    fn=process_script_editor,
                    inputs=[turns_state, speakers_state, speaker_voices_state,
                           editor_language, editor_speed, editor_pause, editor_normalize, editor_translate],
                    outputs=[editor_audio, editor_audio_file, editor_info, raw_preview]
                )
                
                gr.Examples(
                    examples=[
                        [{"speaker": "Speaker 1", "text": "Hello! Welcome to our demo.", "pause": None},
                         {"speaker": "Speaker 2", "text": "This is the script editor!", "pause": None},
                         {"speaker": "Speaker 1", "text": "You can add turns and assign voices.", "pause": 0.5}],
                        [{"speaker": "Host", "text": "Welcome back to the show!", "pause": None},
                         {"speaker": "Guest", "text": "Thanks for having me!", "pause": 0.3},
                         {"speaker": "Host", "text": "Let's talk about your new project.", "pause": None}],
                    ],
                    inputs=[turns_state]
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
        - **Raw Text:** Paste dialogue in `Speaker: text` format
        - **Script Editor:** Visual builder with colored turns
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
