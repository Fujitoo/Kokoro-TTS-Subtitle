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


def create_speaker_card(speaker_id, speaker_name="", selected_voice="af_heart"):
    """Create a speaker card with name input and voice dropdown."""
    with gr.Row(variant="panel", elem_classes="speaker-card"):
        with gr.Column(scale=1):
            name_input = gr.Textbox(
                label="Speaker Name",
                value=speaker_name,
                placeholder=f"Speaker {speaker_id}",
                elem_id=f"speaker_name_{speaker_id}"
            )
        with gr.Column(scale=2):
            voice_dropdown = gr.Dropdown(
                label="Voice",
                choices=ALL_VOICES,
                value=selected_voice,
                elem_id=f"speaker_voice_{speaker_id}"
            )
        with gr.Column(scale=0.5, min_width=100):
            remove_btn = gr.Button("Remove", variant="secondary", size="sm")
    return name_input, voice_dropdown, remove_btn


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


def update_speaker_cards(script_text, current_speakers):
    """Update speaker cards based on script text."""
    script_lines = parse_script(script_text)
    unique_speakers = get_unique_speakers(script_lines)
    
    # Only update if speakers changed
    if len(unique_speakers) == len(current_speakers):
        if all(unique_speakers[i] == current_speakers[i] for i in range(len(unique_speakers))):
            return [gr.update() for _ in range(len(current_speakers) * 3)]
    
    # Build new speaker cards
    updates = []
    for i, speaker_name in enumerate(unique_speakers):
        updates.extend([
            gr.update(value=speaker_name),  # name input
            gr.update(value="af_heart"),     # voice dropdown
            gr.update(visible=True)          # remove button
        ])
    
    # Hide extra cards if any
    remaining = max(0, len(current_speakers) - len(unique_speakers))
    for _ in range(remaining):
        updates.extend([
            gr.update(value=""),
            gr.update(value="af_heart"),
            gr.update(visible=False)
        ])
    
    return updates


def add_speaker(speaker_list):
    """Add a new speaker to the list."""
    new_id = len(speaker_list) + 1
    speaker_list.append({"id": new_id, "name": f"Speaker {new_id}", "voice": "af_heart"})
    return speaker_list, speaker_list


def remove_speaker(speaker_list, index):
    """Remove a speaker at the given index."""
    if 0 <= index < len(speaker_list):
        speaker_list.pop(index)
        # Re-index
        for i, speaker in enumerate(speaker_list):
            speaker["id"] = i + 1
    return speaker_list, speaker_list


def generate_single_audio(text, voice, speaker_name):
    """Generate audio for single speaker mode (placeholder)."""
    if not text.strip():
        return None
    # Placeholder - integrate with actual Kokoro TTS
    return None


def generate_multi_audio(script_text, speaker_list):
    """Generate audio for multi speaker mode (placeholder)."""
    if not script_text.strip():
        return None
    # Placeholder - integrate with actual Kokoro TTS
    return None


def render_speaker_cards(speaker_list):
    """Render speaker cards from speaker list."""
    cards = []
    for speaker in speaker_list:
        with gr.Group():
            name, voice, remove = create_speaker_card(
                speaker["id"],
                speaker["name"],
                speaker["voice"]
            )
            cards.append((name, voice, remove, speaker["id"]))
    return cards


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
            speaker_state = gr.State([
                {"id": 1, "name": "Speaker 1", "voice": "af_heart"},
                {"id": 2, "name": "Speaker 2", "voice": "af_bella"}
            ])
            
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
                    
                    # Container for speaker cards
                    with gr.Group() as speaker_container:
                        # Initial speaker cards
                        speaker_cards = []
                        for i in range(2):
                            with gr.Group():
                                name, voice, remove = create_speaker_card(i + 1)
                                speaker_cards.append((name, voice, remove, i + 1))
                    
                    with gr.Row():
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
            
            # Event handlers
            def update_multi_speakers(speaker_list):
                """Update the speaker cards UI based on speaker list."""
                return speaker_list
            
            add_speaker_btn.click(
                fn=add_speaker,
                inputs=[speaker_state],
                outputs=[speaker_state, speaker_state]
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
