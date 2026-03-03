"""
Kokoro TTS Gradio UI - Dark Theme
Inspired by Google AI Studio's speech playground layout
"""

import gradio as gr

# All available Kokoro voices organized by category
VOICE_OPTIONS = {
    "American Female": ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_sky", "af_sarah", "af_nova", "af_river"],
    "American Male": ["am_adam", "am_michael", "am_echo", "am_eric", "am_liam", "am_onyx"],
    "British Female": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"],
    "British Male": ["bm_george", "bm_lewis", "bm_daniel", "bm_fable"]
}

ALL_VOICES = []
for category, voices in VOICE_OPTIONS.items():
    ALL_VOICES.extend(voices)

MAX_SPEAKERS = 10


def parse_script(script_text):
    """Parse the script text into speaker lines."""
    if not script_text.strip():
        return []
    
    lines = []
    for line in script_text.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            lines.append({"speaker": speaker, "text": text})
    return lines


def get_unique_speakers(script_lines):
    """Extract unique speakers from script lines in order of appearance."""
    seen = set()
    unique_speakers = []
    for line in script_lines:
        speaker = line["speaker"]
        if speaker not in seen:
            seen.add(speaker)
            unique_speakers.append(speaker)
    return unique_speakers


def build_speaker_data(script_text, current_data):
    """Build speaker data from script, preserving existing voice assignments."""
    script_lines = parse_script(script_text)
    unique_speakers = get_unique_speakers(script_lines)
    
    # Create new speaker list, preserving voices for existing speakers
    new_data = []
    for name in unique_speakers[:MAX_SPEAKERS]:
        # Find existing voice for this speaker
        voice = "af_heart"
        for existing in current_data:
            if existing["name"] == name:
                voice = existing["voice"]
                break
        new_data.append({"name": name, "voice": voice})
    
    return new_data


def update_speaker_ui(script_text, speaker_data):
    """Update speaker cards visibility and values based on script."""
    new_data = build_speaker_data(script_text, speaker_data)
    
    # Build updates for all MAX_SPEAKERS cards
    updates = []
    for i in range(MAX_SPEAKERS):
        if i < len(new_data):
            updates.extend([
                gr.update(value=new_data[i]["name"], visible=True),  # name
                gr.update(value=new_data[i]["voice"], visible=True),  # voice
                gr.update(visible=True)  # remove btn
            ])
        else:
            updates.extend([
                gr.update(value="", visible=False),
                gr.update(value="af_heart", visible=False),
                gr.update(visible=False)
            ])
    
    return new_data, updates


def add_speaker(speaker_data):
    """Add a new empty speaker slot."""
    if len(speaker_data) < MAX_SPEAKERS:
        new_num = len(speaker_data) + 1
        speaker_data.append({"name": f"Speaker {new_num}", "voice": "af_heart"})
    return speaker_data, build_ui_updates(speaker_data)


def remove_speaker(speaker_data, index):
    """Remove a speaker at the given index."""
    if 0 <= index < len(speaker_data):
        speaker_data.pop(index)
    return speaker_data, build_ui_updates(speaker_data)


def build_ui_updates(speaker_data):
    """Build UI updates for all speaker cards."""
    updates = []
    for i in range(MAX_SPEAKERS):
        if i < len(speaker_data):
            updates.extend([
                gr.update(value=speaker_data[i]["name"], visible=True),
                gr.update(value=speaker_data[i]["voice"], visible=True),
                gr.update(visible=True)
            ])
        else:
            updates.extend([
                gr.update(value="", visible=False),
                gr.update(value="af_heart", visible=False),
                gr.update(visible=False)
            ])
    return updates


def update_speaker_name(speaker_data, index, new_name):
    """Update a speaker's name."""
    if 0 <= index < len(speaker_data):
        speaker_data[index]["name"] = new_name
    return speaker_data


def update_speaker_voice(speaker_data, index, new_voice):
    """Update a speaker's voice."""
    if 0 <= index < len(speaker_data):
        speaker_data[index]["voice"] = new_voice
    return speaker_data


def generate_single_audio(text, voice, speaker_name):
    """Generate audio for single speaker mode."""
    if not text.strip():
        return None
    # TODO: Integrate with actual Kokoro TTS
    print(f"Single Speaker - Voice: {voice}, Text: {text[:50]}...")
    return None


def generate_multi_audio(script_text, speaker_data):
    """Generate audio for multi speaker mode."""
    if not script_text.strip():
        return None
    # TODO: Integrate with actual Kokoro TTS
    print(f"Multi Speaker - {len(speaker_data)} speakers:")
    for s in speaker_data:
        print(f"  {s['name']}: {s['voice']}")
    print(f"Script: {script_text[:100]}...")
    return None


# Build the Gradio UI
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#1a1a2e",
        body_background_fill_dark="#1a1a2e",
        block_background_fill="#16213e",
        block_background_fill_dark="#16213e",
        block_label_background_fill="#0f3460",
        block_label_background_fill_dark="#0f3460",
        block_title_background_fill="#0f3460",
        block_title_background_fill_dark="#0f3460",
        input_background_fill="#1a1a2e",
        input_background_fill_dark="#1a1a2e",
        button_primary_background_fill="#e94560",
        button_primary_background_fill_dark="#e94560",
        button_primary_text_color="white",
        button_secondary_background_fill="#0f3460",
        button_secondary_background_fill_dark="#0f3460",
        body_text_color="#eaeaea",
        body_text_color_dark="#eaeaea",
    ),
    title="Kokoro TTS Studio",
    css="""
    .gradio-container {
        max-width: 1400px !important;
    }
    .speaker-card {
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .tab-nav button {
        font-size: 16px;
        padding: 12px 24px;
    }
    .generate-btn {
        font-size: 16px;
        font-weight: bold;
        padding: 12px 32px;
    }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🎙️ Kokoro TTS Studio
        Text-to-Speech synthesis with multiple voices and speakers
        """,
        elem_classes="header"
    )
    
    with gr.Tabs() as tabs:
        
        # ==================== SINGLE SPEAKER TAB ====================
        with gr.TabItem("🎤 Single Speaker", id="single"):
            with gr.Row():
                with gr.Column(scale=2):
                    single_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to convert to speech...",
                        lines=8,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            single_voice = gr.Dropdown(
                                label="Voice",
                                choices=ALL_VOICES,
                                value="af_heart",
                                info="Select a voice from available options"
                            )
                        with gr.Column(scale=1):
                            single_speaker = gr.Textbox(
                                label="Speaker Name",
                                placeholder="Speaker 1",
                                info="Optional speaker name"
                            )
                    
                    single_generate = gr.Button(
                        "🎵 Generate Audio",
                        variant="primary",
                        size="lg",
                        elem_classes="generate-btn"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🔊 Preview")
                    single_audio = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        show_download_button=True
                    )
            
            single_generate.click(
                fn=generate_single_audio,
                inputs=[single_text, single_voice, single_speaker],
                outputs=[single_audio]
            )
        
        # ==================== MULTI SPEAKER TAB ====================
        with gr.TabItem("👥 Multi Speaker", id="multi"):
            # State to track speakers
            speaker_state = gr.State([])
            
            # Script input area
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Script")
                    multi_script = gr.Textbox(
                        label="Script",
                        placeholder="Speaker 1: Hello there\nSpeaker 2: Hi, how are you?\nSpeaker 1: I'm doing great!",
                        lines=15,
                        show_copy_button=True,
                        info="Format: Speaker Name: Dialogue"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Voice Settings")
                    
                    # Fixed speaker cards (MAX_SPEAKERS)
                    speaker_cards_ui = []
                    for i in range(MAX_SPEAKERS):
                        with gr.Group(visible=(i < 2)) as card_group:
                            with gr.Row(variant="panel", elem_classes="speaker-card"):
                                with gr.Column(scale=1):
                                    name_input = gr.Textbox(
                                        label="Speaker Name",
                                        value=f"Speaker {i+1}" if i < 2 else "",
                                        placeholder=f"Speaker {i+1}",
                                        visible=(i < 2)
                                    )
                                with gr.Column(scale=2):
                                    voice_dropdown = gr.Dropdown(
                                        label="Voice",
                                        choices=ALL_VOICES,
                                        value="af_heart",
                                        visible=(i < 2)
                                    )
                                with gr.Column(scale=0.5, min_width=100):
                                    remove_btn = gr.Button(
                                        "🗑️",
                                        variant="secondary",
                                        size="sm",
                                        visible=(i < 2)
                                    )
                            speaker_cards_ui.append({
                                "group": card_group,
                                "name": name_input,
                                "voice": voice_dropdown,
                                "remove": remove_btn
                            })
                    
                    add_speaker_btn = gr.Button(
                        "➕ Add Speaker",
                        variant="secondary",
                        size="sm"
                    )
            
            # Generate button and audio output
            with gr.Row():
                multi_generate = gr.Button(
                    "🎵 Generate Audio",
                    variant="primary",
                    size="lg",
                    elem_classes="generate-btn"
                )
            
            multi_audio = gr.Audio(
                label="Generated Audio",
                type="filepath",
                show_download_button=True
            )
            
            # Initialize speaker state on load
            def init_speakers():
                initial = [
                    {"name": "Speaker 1", "voice": "af_heart"},
                    {"name": "Speaker 2", "voice": "af_bella"}
                ]
                return initial
            
            demo.load(
                fn=init_speakers,
                outputs=[speaker_state]
            )
            
            # Update UI when script changes
            multi_script.change(
                fn=update_speaker_ui,
                inputs=[multi_script, speaker_state],
                outputs=[speaker_state] + [comp for i in range(MAX_SPEAKERS) for comp in [
                    speaker_cards_ui[i]["name"],
                    speaker_cards_ui[i]["voice"],
                    speaker_cards_ui[i]["remove"]
                ]]
            )
            
            # Add speaker button
            add_speaker_btn.click(
                fn=add_speaker,
                inputs=[speaker_state],
                outputs=[speaker_state] + [comp for i in range(MAX_SPEAKERS) for comp in [
                    speaker_cards_ui[i]["name"],
                    speaker_cards_ui[i]["voice"],
                    speaker_cards_ui[i]["remove"]
                ]]
            )
            
            # Wire up name/voice changes and remove buttons
            for i in range(MAX_SPEAKERS):
                # Update name
                speaker_cards_ui[i]["name"].change(
                    fn=update_speaker_name,
                    inputs=[speaker_state, gr.Number(value=i, visible=False), speaker_cards_ui[i]["name"]],
                    outputs=[speaker_state]
                )
                # Update voice
                speaker_cards_ui[i]["voice"].change(
                    fn=update_speaker_voice,
                    inputs=[speaker_state, gr.Number(value=i, visible=False), speaker_cards_ui[i]["voice"]],
                    outputs=[speaker_state]
                )
                # Remove speaker
                speaker_cards_ui[i]["remove"].click(
                    fn=lambda data, idx=i: remove_speaker(data, idx),
                    inputs=[speaker_state],
                    outputs=[speaker_state] + [comp for j in range(MAX_SPEAKERS) for comp in [
                        speaker_cards_ui[j]["name"],
                        speaker_cards_ui[j]["voice"],
                        speaker_cards_ui[j]["remove"]
                    ]]
                )
            
            multi_generate.click(
                fn=generate_multi_audio,
                inputs=[multi_script, speaker_state],
                outputs=[multi_audio]
            )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Kokoro TTS Studio** | Built with Gradio
        """,
        elem_classes="footer"
    )


if __name__ == "__main__":
    demo.launch()
