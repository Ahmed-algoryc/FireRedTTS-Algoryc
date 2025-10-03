#!/usr/bin/env python3
"""
Chat Voice Interface with FireRedTTS2
Complete chat interface with voice cloning, selection, and real-time TTS
"""

import os
import sys
import torch
import torchaudio
import pickle
import gradio as gr
import threading
import queue
import time
import json
from typing import List, Dict, Optional
from fireredtts2.fireredtts2 import FireRedTTS2
from openai import OpenAI

class VoiceManager:
    def __init__(self):
        self.voice_profiles_dir = "voice_profiles"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.voice_profiles = {}
        self.current_voice = None
        
        # Initialize directories
        os.makedirs(self.voice_profiles_dir, exist_ok=True)
        
        # Load FireRedTTS2 model
        self.load_model()
        
        # Load existing voice profiles
        self.load_voice_profiles()
    
    def load_model(self):
        """Load the FireRedTTS2 model"""
        print("🔄 Loading FireRedTTS2 model...")
        try:
            self.model = FireRedTTS2(
                pretrained_dir="./pretrained_models/FireRedTTS2",
                gen_type="monologue",
                device=self.device,
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def load_voice_profiles(self):
        """Load all voice profiles from directory"""
        self.voice_profiles = {}
        if os.path.exists(self.voice_profiles_dir):
            for filename in os.listdir(self.voice_profiles_dir):
                if filename.endswith('.pkl'):
                    voice_name = filename[:-4]  # Remove .pkl extension
                    try:
                        with open(os.path.join(self.voice_profiles_dir, filename), 'rb') as f:
                            profile = pickle.load(f)
                        self.voice_profiles[voice_name] = profile
                        print(f"✅ Loaded voice profile: {voice_name}")
                    except Exception as e:
                        print(f"❌ Error loading {filename}: {e}")
    
    def get_voice_list(self):
        """Get list of available voices"""
        return list(self.voice_profiles.keys())
    
    def create_voice_clone(self, audio_file, voice_name, transcription=None):
        """Create a new voice clone from audio file"""
        try:
            # Auto-transcribe if no transcription provided
            if not transcription:
                transcription = self.auto_transcribe(audio_file)
            
            if not transcription:
                return False, "Failed to transcribe audio. Please provide transcription manually."
            
            # Load and tokenize audio
            audio_tensor = self.model.load_prompt_audio(audio_file)
            audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
            audio_tokens, token_length = self.model._audio_tokenizer.encode(
                audio_tensor.to(self.device),
                audio_length.to(self.device),
                batch_size=48,
            )
            
            # Create voice profile
            voice_profile = {
                'audio_tokens': audio_tokens.cpu(),
                'token_length': token_length.cpu(),
                'transcription': transcription,
                'original_audio_path': audio_file,
                'sample_rate': 16000,
                'original_shape': audio_tensor.shape,
                'extraction_info': {
                    'device_used': self.device,
                    'compression_ratio': audio_tensor.numel() / audio_tokens.numel(),
                    'original_size_mb': audio_tensor.numel() * 4 / 1024 / 1024,
                    'token_size_kb': audio_tokens.numel() * 4 / 1024,
                },
                'model_info': {
                    'pretrained_dir': "./pretrained_models/FireRedTTS2",
                    'gen_type': 'monologue',
                    'device': self.device
                }
            }
            
            # Save voice profile
            profile_path = os.path.join(self.voice_profiles_dir, f"{voice_name}.pkl")
            with open(profile_path, 'wb') as f:
                pickle.dump(voice_profile, f)
            
            # Update in-memory profiles
            self.voice_profiles[voice_name] = voice_profile
            
            return True, f"Voice clone '{voice_name}' created successfully!"
            
        except Exception as e:
            return False, f"Error creating voice clone: {e}"
    
    def auto_transcribe(self, audio_path):
        """Auto-transcribe audio using Whisper"""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def generate_tts(self, text, voice_name):
        """Generate TTS using specified voice"""
        if voice_name not in self.voice_profiles:
            return None, f"Voice '{voice_name}' not found"
        
        try:
            profile = self.voice_profiles[voice_name]
            audio = self.model.generate_monologue(
                text=text,
                prompt_wav=profile['original_audio_path'],
                prompt_text=profile['transcription'],
                temperature=0.8,
                topk=30
            )
            
            # Save temporary audio file
            temp_file = f"temp_tts_{int(time.time())}.wav"
            torchaudio.save(temp_file, audio.cpu(), 24000)
            
            return temp_file, None
            
        except Exception as e:
            return None, f"TTS generation error: {e}"
    
    def generate_tts_chunk(self, text_chunk, voice_name):
        """Generate TTS for a small text chunk"""
        if voice_name not in self.voice_profiles:
            return None, f"Voice '{voice_name}' not found"
        
        try:
            profile = self.voice_profiles[voice_name]
            audio = self.model.generate_monologue(
                text=text_chunk,
                prompt_wav=profile['original_audio_path'],
                prompt_text=profile['transcription'],
                temperature=0.8,
                topk=30
            )
            
            # Save temporary audio file
            temp_file = f"temp_tts_chunk_{int(time.time() * 1000)}.wav"
            torchaudio.save(temp_file, audio.cpu(), 24000)
            
            return temp_file, None
            
        except Exception as e:
            return None, f"TTS generation error: {e}"

class ChatInterface:
    def __init__(self):
        self.voice_manager = VoiceManager()
# OPENAI KEY REMOVED - set via env OPENAI_API_KEY
        self.chat_history = []
    
    def stream_gpt_response(self, prompt):
        """Stream GPT response with 100 character limit"""
        try:
            stream = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True,
                max_tokens=100  # Limit to ~100 characters
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    # Stop at 100 characters
                    if len(full_response) >= 100:
                        full_response = full_response[:100]
                        yield content, full_response
                        break
                    
                    yield content, full_response
            
        except Exception as e:
            yield f"Error: {e}", f"Error: {e}"
    
    def chat_with_tts(self, message, voice_name, history):
        """Chat with GPT and generate streaming TTS response"""
        if not message.strip():
            return history, "", None
        
        # Add user message to history
        history.append([message, None])
        
        # Stream GPT response and generate TTS in chunks
        full_response = ""
        accumulated_text = ""
        chunk_size = 20  # Generate TTS every 20 characters
        
        for chunk, full_text in self.stream_gpt_response(message):
            full_response = full_text
            accumulated_text += chunk
            
            # Update history with streaming response
            history[-1][1] = full_response
            
            # Generate TTS for accumulated chunk if we have enough text
            if voice_name and voice_name != "None" and len(accumulated_text) >= chunk_size:
                audio_file, error = self.voice_manager.generate_tts_chunk(accumulated_text, voice_name)
                accumulated_text = ""  # Reset accumulated text
                
                if audio_file:
                    yield history, "", audio_file
                else:
                    yield history, f"TTS Error: {error}", None
            else:
                yield history, "", None
        
        # Generate TTS for any remaining text
        if voice_name and voice_name != "None" and accumulated_text.strip():
            audio_file, error = self.voice_manager.generate_tts_chunk(accumulated_text, voice_name)
            if audio_file:
                yield history, "", audio_file
            else:
                yield history, f"TTS Error: {error}", None
    
    def create_voice_clone_interface(self, audio_file, voice_name, transcription):
        """Interface for creating voice clones"""
        if not audio_file:
            return "Please upload an audio file", gr.update(choices=self.voice_manager.get_voice_list())
        
        if not voice_name:
            return "Please enter a voice name", gr.update(choices=self.voice_manager.get_voice_list())
        
        success, message = self.voice_manager.create_voice_clone(audio_file, voice_name, transcription)
        
        # Update voice dropdown
        voice_choices = self.voice_manager.get_voice_list()
        
        return message, gr.update(choices=voice_choices, value=voice_name if success else None), gr.update(choices=voice_choices)

def create_interface():
    """Create the Gradio interface"""
    chat_interface = ChatInterface()
    
    with gr.Blocks(title="Chat Voice Interface", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎭 Chat Voice Interface with FireRedTTS2")
        gr.Markdown("Chat with AI and get responses in your cloned voice!")
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=400, label="Chat History", type="tuples")
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            lines=2
                        )
                        
                        with gr.Row():
                            voice_dropdown = gr.Dropdown(
                                choices=chat_interface.voice_manager.get_voice_list(),
                                label="Select Voice",
                                value=None,
                                allow_custom_value=False
                            )
                            submit_btn = gr.Button("Send", variant="primary")
                        
                        audio_output = gr.Audio(label="AI Response Audio", type="filepath")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 Voice Settings")
                        gr.Markdown("Select a voice to hear AI responses in that voice.")
                        
                        # Voice info display
                        voice_info = gr.Markdown("No voice selected")
                        
                        def update_voice_info(voice_name):
                            if voice_name and voice_name in chat_interface.voice_manager.voice_profiles:
                                profile = chat_interface.voice_manager.voice_profiles[voice_name]
                                duration = profile['extraction_info'].get('duration_seconds', 'Unknown')
                                if duration != 'Unknown':
                                    duration_str = f"{duration:.2f}s"
                                else:
                                    duration_str = "Unknown"
                                
                                info = f"""
                                **Voice: {voice_name}**
                                - Duration: {duration_str}
                                - Compression: {profile['extraction_info']['compression_ratio']:.1f}x
                                - Transcription: {profile['transcription'][:50]}...
                                """
                                return info
                            return "No voice selected"
                        
                        voice_dropdown.change(update_voice_info, voice_dropdown, voice_info)
                
                # Chat functionality
                def chat_fn(message, voice_name, history):
                    for result in chat_interface.chat_with_tts(message, voice_name, history):
                        yield result
                
                submit_btn.click(
                    chat_fn,
                    inputs=[msg, voice_dropdown, chatbot],
                    outputs=[chatbot, msg, audio_output]
                )
                
                msg.submit(
                    chat_fn,
                    inputs=[msg, voice_dropdown, chatbot],
                    outputs=[chatbot, msg, audio_output]
                )
            
            # Voice Cloning Tab
            with gr.Tab("🎭 Voice Cloning"):
                gr.Markdown("### Create a New Voice Clone")
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Upload Audio File (WAV, MP3, etc.)",
                            type="filepath"
                        )
                        
                        voice_name_input = gr.Textbox(
                            label="Voice Name (Choose a unique name for your voice clone)",
                            placeholder="Enter a name for this voice (e.g., 'my_voice')"
                        )
                        
                        transcription_input = gr.Textbox(
                            label="Transcription (Optional - Leave empty for auto-transcription)",
                            placeholder="Enter what's spoken in the audio file...",
                            lines=3
                        )
                        
                        with gr.Row():
                            clone_btn = gr.Button("Create Voice Clone", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")
                        
                        clone_status = gr.Markdown("Ready to create voice clone")
                
                with gr.Column():
                    gr.Markdown("### Available Voices")
                    voice_list = gr.Dropdown(
                        choices=chat_interface.voice_manager.get_voice_list(),
                        label="Current Voices",
                        interactive=False
                    )
                
                # Voice cloning functionality
                def create_voice_clone_fn(audio_file, voice_name, transcription):
                    return chat_interface.create_voice_clone_interface(audio_file, voice_name, transcription)
                
                clone_btn.click(
                    create_voice_clone_fn,
                    inputs=[audio_input, voice_name_input, transcription_input],
                    outputs=[clone_status, voice_dropdown, voice_list]
                )
                
                clear_btn.click(
                    lambda: (None, "", "", "Form cleared"),
                    outputs=[audio_input, voice_name_input, transcription_input, clone_status]
                )
            
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### System Information")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""
                        **Device:** {chat_interface.voice_manager.device}
                        **Available Voices:** {len(chat_interface.voice_manager.voice_profiles)}
                        **Model Status:** ✅ Loaded
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Features:**
                        - Real-time voice cloning
                        - Streaming GPT responses
                        - Real-time TTS generation
                        - Voice profile management
                        """)
                
                # Voice management
                gr.Markdown("### Voice Management")
                
                def refresh_voices():
                    chat_interface.voice_manager.load_voice_profiles()
                    return gr.update(choices=chat_interface.voice_manager.get_voice_list())
                
                refresh_btn = gr.Button("Refresh Voices")
                refresh_btn.click(refresh_voices, outputs=[voice_dropdown])
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Chat Voice Interface...")
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
