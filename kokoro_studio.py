"""
Kokoro TTS Studio - A Google AI Studio-inspired Gradio UI
Dark-themed interface with three tabs:
1. Single Speaker
2. Multi Speaker – Raw Text
3. Multi Speaker – Script Editor

Full-featured with JavaScript for dynamic speaker/turn management.
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

ALL_VOICES = []
for category, voices in VOICE_CATEGORIES.items():
    ALL_VOICES.extend(voices)

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

def get_all_voices():
    """Fetch all available voices from Hugging Face."""
    try:
        voices = sorted([
            os.path.splitext(file.replace("voices/", ""))[0]
            for file in list_repo_files("hexgrad/Kokoro-82M")
            if file.startswith("voices/")
        ])
        return voices if voices else ALL_VOICES
    except Exception:
        return ALL_VOICES

# ==================== Audio Generation ====================

def update_pipeline(language):
    """Update pipeline if language changed."""
    global pipeline, last_used_language
    lang_code = LANGUAGE_MAP.get(language, "a")
    
    if lang_code != last_used_language:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            last_used_language = lang_code
        except Exception:
            gr.Warning(f"Fallback to English for {language}")
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"

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
    text = clean_text(text)
    lang_code = LANGUAGE_MAP.get(language, "a")
    
    global pipeline, last_used_language
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
        
        # Try different formats
        match = re.match(r'^([^:]+):\s*(.+)$', line)
        if match:
            speaker = match.group(1).strip()
            text = match.group(2).strip()
            speakers.add(speaker)
            lines.append({"speaker": speaker, "text": text, "pause": None})
        else:
            # Continuation of previous speaker
            if lines:
                lines[-1]["text"] += " " + line
    
    return lines, list(speakers)

def create_speaker_cards_html(speakers, speaker_voices):
    """Create HTML for speaker cards."""
    cards = []
    for speaker in speakers:
        current_voice = speaker_voices.get(speaker, "af_bella")
        
        # Build voice options with optgroups
        voice_options = ""
        for category, voices in VOICE_CATEGORIES.items():
            voice_options += f'<optgroup label="{category}">'
            for voice in voices:
                selected = 'selected' if voice == current_voice else ''
                voice_options += f'<option value="{voice}" {selected}>{voice}</option>'
            voice_options += '</optgroup>'
        
        card = f"""
        <div class="speaker-card" data-speaker="{speaker}">
            <div class="speaker-card-header">
                <input type="text" class="speaker-name-input" value="{speaker}" 
                       placeholder="Speaker Name" data-original="{speaker}">
                <button class="remove-speaker-btn" onclick="removeRawSpeaker('{speaker}')" data-speaker="{speaker}">🗑️ Remove</button>
            </div>
            <div class="speaker-voice-select">
                <label style="color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; display: block;">Voice:</label>
                <select class="voice-dropdown" data-speaker="{speaker}" onchange="updateRawSpeakerVoice('{speaker}', this.value)">
                    {voice_options}
                </select>
            </div>
        </div>
        """
        cards.append(card)
    
    return ''.join(cards) if cards else '<p class="no-speakers">No speakers detected. Add text in the format "Speaker Name: dialogue"</p>'

def process_raw_text_multispeaker(script_text, speaker_voices_json, language, speed, pause_duration, normalize_audio, auto_translate):
    """Process raw text multi-speaker script."""
    if not script_text.strip():
        gr.Error("Please enter script text")
        return None, None, "No script text entered"
    
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    # Parse script
    script_lines, detected_speakers = parse_raw_script(script_text)
    
    if not script_lines:
        gr.Error("Could not parse script. Use format: Speaker Name: dialogue")
        return None, None, "❌ Could not parse script"
    
    # Update speaker voices from cards
    for line in script_lines:
        speaker = line["speaker"]
        if speaker not in speaker_voices:
            speaker_voices[speaker] = "af_bella"
    
    # Translate if requested
    if auto_translate:
        try:
            lang_code = LANGUAGE_MAP_LOCAL.get(language, "en")
            for line in script_lines:
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
        return output_path, output_path, info
    return None, None, "❌ Audio generation failed"

# ==================== Multi Speaker - Script Editor Tab ====================

def generate_raw_preview(turns_json):
    """Generate raw script preview from turns."""
    turns = json.loads(turns_json) if turns_json else []
    
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Speaker")
        text = turn.get("text", "")
        pause = turn.get("pause")
        
        line = f"{speaker}: {text}"
        if pause is not None:
            lines.append(line)
            lines.append(f"{{pause: {pause}}}")
        else:
            lines.append(line)
    
    return "\n".join(lines)

def process_script_editor_multispeaker(turns_json, speakers_json, speaker_voices_json, 
                                        language, speed, pause_duration, normalize_audio, auto_translate):
    """Process script editor multi-speaker."""
    turns = json.loads(turns_json) if turns_json else []
    speakers = json.loads(speakers_json) if speakers_json else {}
    speaker_voices = json.loads(speaker_voices_json) if speaker_voices_json else {}
    
    script_lines = []
    for turn in turns:
        script_lines.append({
            "speaker": turn.get("speaker", "Speaker 1"),
            "text": turn.get("text", ""),
            "pause": turn.get("pause")
        })
    
    if not script_lines:
        gr.Error("Please add at least one turn")
        return None, None, "No turns added", ""
    
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
    raw_preview = generate_raw_preview(turns_json)
    
    if output_path:
        return output_path, output_path, info, raw_preview
    return None, None, "❌ Audio generation failed", raw_preview

# ==================== JavaScript for Dynamic UI ====================

def get_raw_text_js():
    """JavaScript for Raw Text tab speaker management."""
    return """
<script>
function removeRawSpeaker(speaker) {
    console.log('Remove speaker:', speaker);
    // Find and remove the card
    const card = document.querySelector(`.speaker-card[data-speaker="${speaker}"]`);
    if (card) {
        card.remove();
    }
}

function updateRawSpeakerVoice(speaker, voice) {
    console.log('Update voice for', speaker, ':', voice);
    // Store in a data attribute for Gradio to read
    const card = document.querySelector(`.speaker-card[data-speaker="${speaker}"]`);
    if (card) {
        card.dataset.voice = voice;
    }
}
</script>
"""

def get_script_editor_js():
    """JavaScript for Script Editor tab turn/speaker management."""
    return """
<script>
let editorTurns = [];
let editorSpeakers = {};
let editorSpeakerVoices = {};
let turnCounter = 0;

const colors = ["#4285f4", "#34a853", "#fbbc04", "#ea4335", "#9c27b0", "#00bcd4", "#ff5722"];

function getVoiceOptions(selectedVoice) {
    const categories = {
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
        "Mandarin Chinese": ["zf_xiaoni", "zm_yunjian"]
    };
    
    let options = '';
    for (const [category, voices] of Object.entries(categories)) {
        options += `<optgroup label="${category}">`;
        for (const voice of voices) {
            const selected = voice === selectedVoice ? 'selected' : '';
            options += `<option value="${voice}" ${selected}>${voice}</option>`;
        }
        options += '</optgroup>';
    }
    return options;
}

function getSpeakerOptions(selectedSpeaker) {
    let options = '';
    for (const speaker of Object.keys(editorSpeakers)) {
        const selected = speaker === selectedSpeaker ? 'selected' : '';
        options += `<option value="${speaker}" ${selected}>${speaker}</option>`;
    }
    return options;
}

function renderTurns() {
    const container = document.querySelector('.turns-container');
    if (!container || editorTurns.length === 0) {
        if (container) {
            container.innerHTML = '<p class="no-speakers">Click "Add Turn" to start building your script</p>';
        }
        updateRawPreview();
        updateSpeakerCards();
        updateGradioState();
        return;
    }
    
    let html = '';
    editorTurns.forEach((turn, index) => {
        const speaker = turn.speaker;
        const speakerIdx = Object.keys(editorSpeakers).indexOf(speaker);
        const color = colors[speakerIdx % colors.length];
        
        const pauseHtml = turn.pause !== null && turn.pause !== undefined 
            ? `<input type="number" class="turn-pause-input" placeholder="Pause (s)" step="0.1" data-turn="${index}" value="${turn.pause}" onchange="updateTurnPause(${index}, this.value)">`
            : `<button class="add-pause-btn" onclick="addPauseToTurn(${index})" style="background: var(--bg-secondary); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 4px 8px; border-radius: 4px; margin-top: 8px; cursor: pointer;">+ Add Pause</button>`;
        
        html += `
        <div class="turn-card" data-turn="${index}">
            <div class="speaker-dot" style="background: ${color};"></div>
            <div class="turn-content">
                <select class="turn-speaker-select" data-turn="${index}" onchange="updateTurnSpeaker(${index}, this.value)">
                    ${getSpeakerOptions(speaker)}
                </select>
                <textarea class="turn-text-input" data-turn="${index}" rows="2" oninput="updateTurnText(${index}, this.value)">${turn.text}</textarea>
                ${pauseHtml}
            </div>
            <button class="remove-turn-btn" onclick="removeTurn(${index})">🗑️</button>
        </div>
        `;
    });
    
    container.innerHTML = html;
    updateRawPreview();
    updateSpeakerCards();
    updateGradioState();
}

function addTurn() {
    // Find next speaker number
    let speakerNum = 1;
    while (editorSpeakers[`Speaker ${speakerNum}`]) {
        speakerNum++;
    }
    const newSpeaker = `Speaker ${speakerNum}`;
    editorSpeakers[newSpeaker] = true;
    if (!editorSpeakerVoices[newSpeaker]) {
        editorSpeakerVoices[newSpeaker] = 'af_bella';
    }
    
    editorTurns.push({
        id: turnCounter++,
        speaker: newSpeaker,
        text: 'Enter dialogue text here...',
        pause: null
    });
    
    renderTurns();
}

function removeTurn(index) {
    editorTurns.splice(index, 1);
    
    // Remove speaker if no longer used
    const usedSpeakers = new Set(editorTurns.map(t => t.speaker));
    for (const speaker of Object.keys(editorSpeakers)) {
        if (!usedSpeakers.has(speaker)) {
            delete editorSpeakers[speaker];
            delete editorSpeakerVoices[speaker];
        }
    }
    
    renderTurns();
}

function clearAllTurns() {
    editorTurns = [];
    editorSpeakers = {};
    editorSpeakerVoices = {};
    turnCounter = 0;
    renderTurns();
}

function updateTurnSpeaker(index, newSpeaker) {
    editorTurns[index].speaker = newSpeaker;
    renderTurns();
}

function updateTurnText(index, newText) {
    editorTurns[index].text = newText;
    updateRawPreview();
}

function addPauseToTurn(index) {
    editorTurns[index].pause = 0.5;
    renderTurns();
}

function updateTurnPause(index, value) {
    editorTurns[index].pause = value ? parseFloat(value) : null;
    updateRawPreview();
}

function updateRawPreview() {
    const preview = document.querySelector('.raw-preview textarea, .raw-preview');
    if (!preview) return;
    
    let lines = [];
    editorTurns.forEach(turn => {
        lines.push(`${turn.speaker}: ${turn.text}`);
        if (turn.pause !== null && turn.pause !== undefined) {
            lines.push(`{pause: ${turn.pause}}`);
        }
    });
    
    const text = lines.join('\\n');
    if (preview.tagName === 'TEXTAREA') {
        preview.value = text;
    } else {
        preview.textContent = text;
    }
}

function updateSpeakerCards() {
    const container = document.querySelector('.editor-speaker-cards');
    if (!container) return;
    
    if (Object.keys(editorSpeakers).length === 0) {
        container.innerHTML = '<p class="no-speakers">Add turns to see speaker cards</p>';
        return;
    }
    
    let html = '';
    for (const speaker of Object.keys(editorSpeakers)) {
        const currentVoice = editorSpeakerVoices[speaker] || 'af_bella';
        
        html += `
        <div class="speaker-card" data-speaker="${speaker}">
            <div class="speaker-card-header">
                <input type="text" class="speaker-name-input" value="${speaker}" 
                       placeholder="Speaker Name" onchange="renameSpeaker('${speaker}', this.value)">
                <button class="remove-speaker-btn" onclick="removeEditorSpeaker('${speaker}')">🗑️ Remove</button>
            </div>
            <div class="speaker-voice-select">
                <label style="color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; display: block;">Voice:</label>
                <select class="voice-dropdown" data-speaker="${speaker}" onchange="updateSpeakerVoice('${speaker}', this.value)">
                    ${getVoiceOptions(currentVoice)}
                </select>
            </div>
        </div>
        `;
    }
    
    container.innerHTML = html;
}

function renameSpeaker(oldName, newName) {
    if (oldName === newName || !newName.trim()) return;
    
    // Update speaker in turns
    editorTurns.forEach(turn => {
        if (turn.speaker === oldName) {
            turn.speaker = newName;
        }
    });
    
    // Update speaker voices
    editorSpeakerVoices[newName] = editorSpeakerVoices[oldName];
    delete editorSpeakerVoices[oldName];
    
    // Update speakers
    editorSpeakers[newName] = true;
    delete editorSpeakers[oldName];
    
    renderTurns();
}

function removeEditorSpeaker(speaker) {
    // Check if speaker is used
    const isUsed = editorTurns.some(t => t.speaker === speaker);
    if (isUsed) {
        alert('Cannot remove speaker that is used in turns. Remove or reassign the turns first.');
        return;
    }
    
    delete editorSpeakers[speaker];
    delete editorSpeakerVoices[speaker];
    updateSpeakerCards();
    updateGradioState();
}

function updateSpeakerVoice(speaker, voice) {
    editorSpeakerVoices[speaker] = voice;
    updateGradioState();
}

function addEditorSpeaker() {
    const input = document.querySelector('.editor-add-speaker-name');
    if (!input) return;
    
    const name = input.value.trim();
    if (!name) {
        alert('Please enter a speaker name');
        return;
    }
    
    if (editorSpeakers[name]) {
        alert('Speaker already exists');
        return;
    }
    
    editorSpeakers[name] = true;
    editorSpeakerVoices[name] = 'af_bella';
    input.value = '';
    updateSpeakerCards();
    updateGradioState();
}

function updateGradioState() {
    // Update hidden JSON states for Gradio
    const turnsState = document.querySelector('.editor-turns-state textarea');
    const speakersState = document.querySelector('.editor-speakers-state textarea');
    const voicesState = document.querySelector('.editor-voices-state textarea');
    
    if (turnsState) turnsState.value = JSON.stringify(editorTurns);
    if (speakersState) speakersState.value = JSON.stringify(editorSpeakers);
    if (voicesState) voicesState.value = JSON.stringify(editorSpeakerVoices);
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Find and attach click handlers
    const addTurnBtn = document.querySelector('.editor-add-turn-btn');
    if (addTurnBtn) {
        addTurnBtn.addEventListener('click', addTurn);
    }
    
    const clearBtn = document.querySelector('.editor-clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearAllTurns);
    }
    
    const addSpeakerBtn = document.querySelector('.editor-add-speaker-btn');
    if (addSpeakerBtn) {
        addSpeakerBtn.addEventListener('click', addEditorSpeaker);
    }
});
</script>

<style>
.editor-speaker-cards {
    max-height: 400px;
    overflow-y: auto;
}
</style>
"""

# ==================== Gradio UI ====================

def create_ui():
    """Create the main Gradio UI."""
    
    # Custom CSS for Google AI Studio-inspired dark theme
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
    
    /* Fix dropdown text color */
    .gr-dropdown .wrap {
        background: var(--bg-card) !important;
    }
    
    .gr-dropdown .wrap input {
        color: var(--text-primary) !important;
    }
    
    select option {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    .gr-markdown {
        color: var(--text-primary) !important;
    }
    
    .gr-accordion {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Kokoro TTS Studio", theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # 🎙️ Kokoro TTS Studio
        
        Professional text-to-speech interface inspired by Google AI Studio
        """)
        
        # Inject JavaScript
        gr.HTML(get_raw_text_js())
        gr.HTML(get_script_editor_js())
        
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
                            choices=[],
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
Speaker 2: That's wonderful to hear.""",
                            lines=15
                        )
                        
                        raw_parse_btn = gr.Button("📋 Parse Script", size="sm")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎭 Speaker Voices")
                        raw_speaker_cards = gr.HTML(
                            value='<p class="no-speakers">Enter script text to see speaker cards</p>',
                            elem_classes=["speaker-cards-container"]
                        )
                        
                        raw_speaker_voices_state = gr.JSON(value={}, visible=False)
                
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
                    raw_normalize = gr.Checkbox(
                        label="🔊 Normalize Speaker Volumes",
                        value=True
                    )
                    raw_translate = gr.Checkbox(
                        label="🌐 Auto-translate to target language",
                        value=False
                    )
                
                raw_generate = gr.Button(
                    "🚀 Generate Audio",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### 🎧 Output")
                raw_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    autoplay=True
                )
                raw_audio_file = gr.File(label="📥 Download Audio")
                raw_info = gr.Markdown("Click Generate to create multi-speaker audio...")
            
            # ==================== Multi Speaker - Script Editor Tab ====================
            with gr.TabItem("✏️ Multi Speaker – Script Editor", id="multi_editor"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Visual Script Builder")
                        
                        editor_turns_container = gr.HTML(
                            value='<p class="no-speakers">Click "Add Turn" to start building your script</p>',
                            elem_classes=["turns-container"]
                        )
                        
                        with gr.Row():
                            editor_add_turn = gr.Button("➕ Add Turn", size="sm", elem_classes=["editor-add-turn-btn"])
                            editor_remove_all = gr.Button("🗑️ Clear All", size="sm", elem_classes=["editor-clear-btn"])
                        
                        # Hidden states - using Textbox instead of JSON for JS access
                        editor_turns_state = gr.Textbox(value="[]", visible=False, elem_classes=["editor-turns-state"])
                        editor_speakers_state = gr.Textbox(value="{}", visible=False, elem_classes=["editor-speakers-state"])
                        editor_voices_state = gr.Textbox(value="{}", visible=False, elem_classes=["editor-voices-state"])
                        
                        gr.Markdown("### 📄 Raw Script Preview")
                        raw_preview = gr.Textbox(
                            label="",
                            lines=6,
                            interactive=False,
                            elem_classes=["raw-preview"]
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎭 Speaker Voices")
                        editor_speaker_cards = gr.HTML(
                            value='<p class="no-speakers">Add turns to see speaker cards</p>',
                            elem_classes=["speaker-cards-container", "editor-speaker-cards"]
                        )
                        
                        editor_add_speaker = gr.Button("➕ Add Speaker", size="sm", elem_classes=["editor-add-speaker-btn"])
                        editor_speaker_name = gr.Textbox(
                            label="New Speaker Name",
                            placeholder="Enter speaker name",
                            elem_classes=["editor-add-speaker-name"],
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
                    editor_normalize = gr.Checkbox(
                        label="🔊 Normalize Speaker Volumes",
                        value=True
                    )
                    editor_translate = gr.Checkbox(
                        label="🌐 Auto-translate to target language",
                        value=False
                    )
                
                editor_generate = gr.Button(
                    "🚀 Generate Audio",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### 🎧 Output")
                editor_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    autoplay=True
                )
                editor_audio_file = gr.File(label="📥 Download Audio")
                editor_info = gr.Markdown("Click Generate to create multi-speaker audio...")
        
        # ==================== Event Handlers ====================
        
        # Populate voice dropdowns on load
        def load_voices():
            voices = get_all_voices()
            return gr.Dropdown(choices=voices, value="af_bella")
        
        demo.load(load_voices, outputs=[single_voice])
        
        # Single Speaker
        single_generate.click(
            fn=process_single_speaker,
            inputs=[single_text, single_voice, single_speaker_name, single_language, 
                   single_speed, single_translate],
            outputs=[single_audio, single_info]
        )
        
        # Raw Text Multi Speaker - Parse script
        def parse_and_show_cards(script_text, current_voices_json):
            if not script_text.strip():
                return '<p class="no-speakers">Enter script text to see speaker cards</p>', current_voices_json
            
            current_voices = json.loads(current_voices_json) if current_voices_json else {}
            _, speakers = parse_raw_script(script_text)
            
            for speaker in speakers:
                if speaker not in current_voices:
                    current_voices[speaker] = "af_bella"
            
            cards_html = create_speaker_cards_html(speakers, current_voices)
            return cards_html, json.dumps(current_voices)
        
        raw_parse_btn.click(
            fn=parse_and_show_cards,
            inputs=[raw_script, raw_speaker_voices_state],
            outputs=[raw_speaker_cards, raw_speaker_voices_state]
        )
        
        raw_script.change(
            fn=parse_and_show_cards,
            inputs=[raw_script, raw_speaker_voices_state],
            outputs=[raw_speaker_cards, raw_speaker_voices_state]
        )
        
        # Raw Text Multi Speaker - Generate
        raw_generate.click(
            fn=process_raw_text_multispeaker,
            inputs=[raw_script, raw_speaker_voices_state, raw_language, raw_speed, 
                   raw_pause, raw_normalize, raw_translate],
            outputs=[raw_audio, raw_audio_file, raw_info]
        )
        
        # Script Editor - Generate
        def process_editor(turns_text, speakers_text, voices_text, language, speed, pause, normalize, translate):
            return process_script_editor_multispeaker(
                turns_text, speakers_text, voices_text,
                language, speed, pause, normalize, translate
            )
        
        editor_generate.click(
            fn=process_editor,
            inputs=[editor_turns_state, editor_speakers_state, editor_voices_state,
                   editor_language, editor_speed, editor_pause, editor_normalize, editor_translate],
            outputs=[editor_audio, editor_audio_file, editor_info, raw_preview]
        )
    
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
