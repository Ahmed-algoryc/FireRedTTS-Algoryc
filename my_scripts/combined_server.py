#!/usr/bin/env python3
"""
Combined HTTP + WebSocket Server for Real-time Voice Chat
"""

import asyncio
import websockets
import json
import base64
import io
import torch
import torchaudio
import pickle
import os
import threading
import time
import http.server
import socketserver
import whisper
from fireredtts2.fireredtts2 import FireRedTTS2
from openai import OpenAI

class WebSocketTTSServer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.whisper_model = None
        self.voice_profiles = {}
# OPENAI KEY REMOVED - set via env OPENAI_API_KEY
        
        # Load model and voice profiles
        self.load_model()
        self.load_whisper_model()
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
    
    def load_whisper_model(self):
        """Load Whisper model for auto-transcription"""
        print("🔄 Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            self.whisper_model = None
    
    def load_voice_profiles(self):
        """Load all voice profiles"""
        self.voice_profiles = {}
        voice_profiles_dir = "voice_profiles"
        if os.path.exists(voice_profiles_dir):
            for filename in os.listdir(voice_profiles_dir):
                if filename.endswith('.pkl'):
                    voice_name = filename[:-4]
                    try:
                        with open(os.path.join(voice_profiles_dir, filename), 'rb') as f:
                            profile = pickle.load(f)
                        self.voice_profiles[voice_name] = profile
                        print(f"✅ Loaded voice profile: {voice_name}")
                    except Exception as e:
                        print(f"❌ Error loading {filename}: {e}")
    
    def get_voice_list(self):
        """Get list of available voices"""
        return list(self.voice_profiles.keys())
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            return "Transcription not available (Whisper model not loaded)"
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            return f"Transcription failed: {str(e)}"
    
    def generate_tts_chunk(self, text_chunk, voice_name):
        """Generate TTS for a text chunk and return as base64 audio (NO FILE SAVING)"""
        if voice_name not in self.voice_profiles:
            return None, f"Voice '{voice_name}' not found"
        
        try:
            profile = self.voice_profiles[voice_name]
            
            # Generate audio directly in memory - NO FILE OPERATIONS
            audio = self.model.generate_monologue(
                text=text_chunk,
                prompt_wav=profile['original_audio_path'],
                prompt_text=profile['transcription'],
                temperature=0.8,
                topk=30
            )
            
            # Convert to bytes in memory only - NO FILE SAVING
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio.cpu(), 24000, format="wav")
            audio_bytes = buffer.getvalue()
            buffer.close()  # Clean up buffer
            
            # Encode as base64 for streaming
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Clear audio tensor from memory
            del audio
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return audio_base64, None
            
        except Exception as e:
            return None, f"TTS generation error: {e}"
    
    async def generate_tts_chunk_async(self, text_chunk, voice_name):
        """Async wrapper for TTS generation"""
        # Run TTS generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_tts_chunk, text_chunk, voice_name)
    
    async def stream_gpt_response(self, prompt):
        """Stream GPT response with 100 character limit"""
        try:
            stream = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True,
                max_tokens=100
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
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        print(f"🔌 Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'chat':
                    # Handle chat request
                    user_message = data.get('message', '')
                    voice_name = data.get('voice', '')
                    
                    if not user_message:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'No message provided'
                        }))
                        continue
                    
                    # Send initial response
                    await websocket.send(json.dumps({
                        'type': 'chat_start',
                        'message': 'Starting response...'
                    }))
                    
                    # Stream GPT response and generate TTS
                    accumulated_text = ""
                    chunk_size = 8  # Smaller chunks for faster TTS generation
                    tts_tasks = []  # Store TTS generation tasks
                    
                    async for chunk, full_text in self.stream_gpt_response(user_message):
                        # Send text chunk immediately
                        await websocket.send(json.dumps({
                            'type': 'text_chunk',
                            'chunk': chunk,
                            'full_text': full_text
                        }))
                        
                        accumulated_text += chunk
                        
                        # Generate TTS for accumulated chunk (non-blocking)
                        if voice_name and voice_name != "None" and len(accumulated_text) >= chunk_size:
                            # Start TTS generation in background
                            tts_task = asyncio.create_task(
                                self.generate_tts_chunk_async(accumulated_text, voice_name)
                            )
                            tts_tasks.append((tts_task, accumulated_text))
                            accumulated_text = ""  # Reset
                    
                    # Generate TTS for any remaining text
                    if voice_name and voice_name != "None" and accumulated_text.strip():
                        tts_task = asyncio.create_task(
                            self.generate_tts_chunk_async(accumulated_text, voice_name)
                        )
                        tts_tasks.append((tts_task, accumulated_text))
                    
                    # Send all TTS chunks as they complete
                    for tts_task, text in tts_tasks:
                        try:
                            audio_base64, error = await tts_task
                            if audio_base64:
                                await websocket.send(json.dumps({
                                    'type': 'audio_chunk',
                                    'audio': audio_base64,
                                    'text': text
                                }))
                            else:
                                await websocket.send(json.dumps({
                                    'type': 'tts_error',
                                    'error': error
                                }))
                        except Exception as e:
                            await websocket.send(json.dumps({
                                'type': 'tts_error',
                                'error': f'TTS generation failed: {str(e)}'
                            }))
                    
                    # Send completion signal
                    await websocket.send(json.dumps({
                        'type': 'chat_complete',
                        'message': 'Response complete'
                    }))
                
                elif message_type == 'clone_voice':
                    # Handle voice cloning
                    audio_base64 = data.get('audio')
                    voice_name = data.get('voice_name', '')
                    transcription = data.get('transcription', '')
                    
                    if not audio_base64 or not voice_name:
                        await websocket.send(json.dumps({
                            'type': 'clone_error',
                            'message': 'Missing audio data or voice name'
                        }))
                        continue
                    
                    try:
                        # Decode audio
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_buffer = io.BytesIO(audio_bytes)
                        
                        # Load audio with torchaudio
                        audio_tensor, sample_rate = torchaudio.load(audio_buffer)
                        
                        # Save temporary audio file
                        temp_path = f"temp_clone_{voice_name}.wav"
                        torchaudio.save(temp_path, audio_tensor, sample_rate)
                        
                        # Auto-transcribe if no transcription provided
                        if not transcription.strip():
                            await websocket.send(json.dumps({
                                'type': 'clone_status',
                                'message': 'Auto-transcribing audio...'
                            }))
                            transcription = self.transcribe_audio(temp_path)
                            await websocket.send(json.dumps({
                                'type': 'clone_status',
                                'message': f'Transcription: "{transcription}"'
                            }))
                        
                        # Create voice profile
                        voice_profile = {
                            'original_audio_path': temp_path,
                            'transcription': transcription,
                            'sample_rate': sample_rate,
                            'duration_seconds': audio_tensor.shape[1] / sample_rate,
                            'created_at': time.time()
                        }
                        
                        # Save voice profile
                        os.makedirs('voice_profiles', exist_ok=True)
                        profile_path = f'voice_profiles/{voice_name}.pkl'
                        with open(profile_path, 'wb') as f:
                            pickle.dump(voice_profile, f)
                        
                        # Add to loaded profiles
                        self.voice_profiles[voice_name] = voice_profile
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                        await websocket.send(json.dumps({
                            'type': 'clone_success',
                            'message': f'Voice "{voice_name}" cloned successfully!',
                            'voice_name': voice_name,
                            'transcription': transcription
                        }))
                        
                    except Exception as e:
                        await websocket.send(json.dumps({
                            'type': 'clone_error',
                            'message': f'Voice cloning failed: {str(e)}'
                        }))
                
                elif message_type == 'get_voices':
                    # Send available voices
                    await websocket.send(json.dumps({
                        'type': 'voices',
                        'voices': self.get_voice_list()
                    }))
                
                elif message_type == 'ping':
                    # Respond to ping
                    await websocket.send(json.dumps({
                        'type': 'pong'
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"❌ Error handling client: {e}")
            try:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))
            except:
                pass

def create_html_interface():
    """Create the HTML interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Voice Chat</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        
        .status {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .status.connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.disconnected {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 10px;
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            margin-bottom: 20px;
            background: #f9f9f9;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: 20%;
        }
        
        .ai-message {
            background: #e9ecef;
            color: #333;
            margin-right: 20%;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .voice-select {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            min-width: 150px;
        }
        
        .message-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        
        .send-button {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .send-button:hover {
            background: #0056b3;
        }
        
        .send-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .audio-controls {
            text-align: center;
            margin-top: 20px;
        }
        
        .play-button {
            padding: 10px 20px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        
        .play-button:hover {
            background: #218838;
        }
        
        .audio-queue {
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
        
        .voice-clone-section {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .voice-clone-section h3 {
            margin-top: 0;
            color: #495057;
        }
        
        .clone-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }
        
        .voice-name-input, .transcription-input {
            flex: 1;
            min-width: 150px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .audio-file-input {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }
        
        .clone-button {
            padding: 8px 16px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .clone-button:hover {
            background: #218838;
        }
        
        .clone-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .clone-status {
            margin-top: 10px;
            padding: 8px;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .clone-status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .clone-status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .clone-status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Real-time Voice Chat</h1>
        
        <div id="status" class="status disconnected">
            Connecting to server...
        </div>
        
        <div class="input-container">
            <select id="voiceSelect" class="voice-select">
                <option value="">Select Voice...</option>
            </select>
            <input type="text" id="messageInput" class="message-input" placeholder="Type your message here..." maxlength="200">
            <button id="sendButton" class="send-button">Send</button>
        </div>
        
        <div class="voice-clone-section">
            <h3>🎭 Clone New Voice</h3>
            <div class="clone-container">
                <input type="text" id="voiceNameInput" placeholder="Voice name (e.g., 'my_voice')" class="voice-name-input">
                <input type="text" id="transcriptionInput" placeholder="What's said in the audio? (optional - will auto-transcribe)" class="transcription-input">
                <input type="file" id="audioFileInput" accept="audio/*" class="audio-file-input">
                <button id="cloneButton" class="clone-button">Clone Voice</button>
            </div>
            <div id="cloneStatus" class="clone-status"></div>
        </div>
        
        <div id="chatContainer" class="chat-container">
            <div class="message ai-message">
                Welcome! Select a voice and start chatting. Your AI responses will be generated in real-time with voice synthesis.
            </div>
        </div>
        
        <div class="audio-controls">
            <button id="playAllButton" class="play-button" style="display: none;">Play All Audio</button>
            <div id="audioQueue" class="audio-queue"></div>
        </div>
    </div>

    <script>
        class VoiceChat {
            constructor() {
                this.ws = null;
                this.audioQueue = [];
                this.isPlaying = false;
                this.currentAudio = null;
                
                this.initializeElements();
                this.connect();
                this.setupEventListeners();
            }
            
            initializeElements() {
                this.statusEl = document.getElementById('status');
                this.voiceSelect = document.getElementById('voiceSelect');
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.chatContainer = document.getElementById('chatContainer');
                this.playAllButton = document.getElementById('playAllButton');
                this.audioQueueEl = document.getElementById('audioQueue');
                
                // Voice cloning elements
                this.voiceNameInput = document.getElementById('voiceNameInput');
                this.transcriptionInput = document.getElementById('transcriptionInput');
                this.audioFileInput = document.getElementById('audioFileInput');
                this.cloneButton = document.getElementById('cloneButton');
                this.cloneStatus = document.getElementById('cloneStatus');
            }
            
            connect() {
                // Use RunPod proxy URL for WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                let wsUrl;
                
                if (window.location.hostname.includes('runpod.net')) {
                    // RunPod proxy - use the same hostname but different port
                    wsUrl = `${protocol}//${window.location.hostname.replace(':8080', ':8765')}`;
                } else {
                    // Local development
                    wsUrl = `${protocol}//${window.location.hostname}:8765`;
                }
                
                console.log('Connecting to WebSocket:', wsUrl);
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.updateStatus('Connected to server', 'connected');
                    this.loadVoices();
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    this.updateStatus('Disconnected from server', 'disconnected');
                    setTimeout(() => this.connect(), 3000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateStatus('Connection error', 'disconnected');
                };
            }
            
            updateStatus(message, type) {
                this.statusEl.textContent = message;
                this.statusEl.className = `status ${type}`;
            }
            
            loadVoices() {
                this.ws.send(JSON.stringify({ type: 'get_voices' }));
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'voices':
                        this.populateVoiceSelect(data.voices);
                        break;
                    case 'chat_start':
                        this.addMessage('AI is thinking...', 'ai-message');
                        break;
                    case 'text_chunk':
                        this.updateLastMessage(data.full_text, 'ai-message');
                        break;
                    case 'audio_chunk':
                        this.addAudioToQueue(data.audio, data.text);
                        break;
                    case 'chat_complete':
                        this.updateLastMessage('Response complete', 'ai-message');
                        break;
                    case 'clone_success':
                        this.showCloneStatus(data.message, 'success');
                        this.loadVoices(); // Refresh voice list
                        this.resetCloneForm();
                        break;
                    case 'clone_error':
                        this.showCloneStatus(data.message, 'error');
                        this.resetCloneForm();
                        break;
                    case 'clone_status':
                        this.showCloneStatus(data.message, 'info');
                        break;
                    case 'error':
                        this.addMessage(`Error: ${data.message}`, 'ai-message');
                        break;
                }
            }
            
            populateVoiceSelect(voices) {
                this.voiceSelect.innerHTML = '<option value="">Select Voice...</option>';
                voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice;
                    option.textContent = voice;
                    this.voiceSelect.appendChild(option);
                });
            }
            
            addMessage(text, className) {
                const messageEl = document.createElement('div');
                messageEl.className = `message ${className}`;
                messageEl.textContent = text;
                this.chatContainer.appendChild(messageEl);
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
                return messageEl;
            }
            
            updateLastMessage(text, className) {
                const messages = this.chatContainer.querySelectorAll('.message');
                const lastMessage = messages[messages.length - 1];
                if (lastMessage && lastMessage.classList.contains(className)) {
                    lastMessage.textContent = text;
                } else {
                    this.addMessage(text, className);
                }
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
            }
            
            addAudioToQueue(audioBase64, text) {
                const audioBlob = this.base64ToBlob(audioBase64, 'audio/wav');
                const audioUrl = URL.createObjectURL(audioBlob);
                
                this.audioQueue.push({
                    url: audioUrl,
                    text: text,
                    audio: new Audio(audioUrl)
                });
                
                this.updateAudioQueue();
                this.playNextAudio();
            }
            
            base64ToBlob(base64, mimeType) {
                const byteCharacters = atob(base64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                return new Blob([byteArray], { type: mimeType });
            }
            
            updateAudioQueue() {
                this.audioQueueEl.textContent = `Audio queue: ${this.audioQueue.length} chunks`;
                this.playAllButton.style.display = this.audioQueue.length > 0 ? 'block' : 'none';
            }
            
            playNextAudio() {
                if (this.isPlaying || this.audioQueue.length === 0) return;
                
                this.isPlaying = true;
                const audioItem = this.audioQueue.shift();
                this.currentAudio = audioItem.audio;
                
                this.currentAudio.onended = () => {
                    URL.revokeObjectURL(audioItem.url);
                    this.isPlaying = false;
                    this.updateAudioQueue();
                    this.playNextAudio();
                };
                
                this.currentAudio.onerror = () => {
                    console.error('Audio playback error');
                    this.isPlaying = false;
                    this.updateAudioQueue();
                    this.playNextAudio();
                };
                
                this.currentAudio.play().catch(error => {
                    console.error('Audio play failed:', error);
                    this.isPlaying = false;
                    this.updateAudioQueue();
                    this.playNextAudio();
                });
            }
            
            showCloneStatus(message, type) {
                this.cloneStatus.textContent = message;
                this.cloneStatus.className = `clone-status ${type}`;
                
                // Clear status after 5 seconds
                setTimeout(() => {
                    this.cloneStatus.textContent = '';
                    this.cloneStatus.className = 'clone-status';
                }, 5000);
            }
            
            resetCloneForm() {
                this.cloneButton.disabled = false;
                this.cloneButton.textContent = 'Clone Voice';
                this.voiceNameInput.value = '';
                this.transcriptionInput.value = '';
                this.audioFileInput.value = '';
            }
            
            async cloneVoice() {
                const voiceName = this.voiceNameInput.value.trim();
                const transcription = this.transcriptionInput.value.trim();
                const audioFile = this.audioFileInput.files[0];
                
                if (!voiceName) {
                    this.showCloneStatus('Please enter a voice name', 'error');
                    return;
                }
                
                if (!audioFile) {
                    this.showCloneStatus('Please select an audio file', 'error');
                    return;
                }
                
                this.cloneButton.disabled = true;
                this.cloneButton.textContent = 'Cloning...';
                
                try {
                    // Convert audio file to base64
                    const audioBase64 = await this.fileToBase64(audioFile);
                    
                    // Send to server
                    this.ws.send(JSON.stringify({
                        type: 'clone_voice',
                        voice_name: voiceName,
                        transcription: transcription,
                        audio: audioBase64
                    }));
                    
                } catch (error) {
                    this.showCloneStatus(`Error: ${error.message}`, 'error');
                    this.cloneButton.disabled = false;
                    this.cloneButton.textContent = 'Clone Voice';
                }
            }
            
            fileToBase64(file) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64 = reader.result.split(',')[1]; // Remove data:audio/...;base64, prefix
                        resolve(base64);
                    };
                    reader.onerror = reject;
                    reader.readAsDataURL(file);
                });
            }
            
            setupEventListeners() {
                this.sendButton.addEventListener('click', () => this.sendMessage());
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') this.sendMessage();
                });
                
                this.cloneButton.addEventListener('click', () => this.cloneVoice());
                
                this.playAllButton.addEventListener('click', () => {
                    if (this.currentAudio) {
                        this.currentAudio.pause();
                        this.currentAudio.currentTime = 0;
                    }
                    this.isPlaying = false;
                    this.playNextAudio();
                });
            }
            
            sendMessage() {
                const message = this.messageInput.value.trim();
                const voice = this.voiceSelect.value;
                
                if (!message) return;
                
                // Add user message to chat
                this.addMessage(message, 'user-message');
                
                // Send to server
                this.ws.send(JSON.stringify({
                    type: 'chat',
                    message: message,
                    voice: voice
                }));
                
                // Clear input
                this.messageInput.value = '';
                this.sendButton.disabled = true;
                
                // Re-enable after a short delay
                setTimeout(() => {
                    this.sendButton.disabled = false;
                }, 1000);
            }
        }
        
        // Initialize the chat when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new VoiceChat();
        });
    </script>
</body>
</html>
    """
    
    with open('voice_chat.html', 'w') as f:
        f.write(html_content)
    
    print("✅ HTML interface created: voice_chat.html")

async def main():
    """Main server function"""
    print("🚀 Starting Combined HTTP + WebSocket Server...")
    
    # Create HTML interface
    create_html_interface()
    
    # Create server instance
    server = WebSocketTTSServer()
    
    print("🌐 WebSocket server starting on ws://0.0.0.0:8765")
    print("📄 HTML interface available at: voice_chat.html")
    print("🔌 Connect to: ws://localhost:8765")
    print("🌐 Open: http://localhost:8080/voice_chat.html")
    
    # Start HTTP server in a separate thread
    def start_http():
        PORT = 8080
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
            print(f"🌐 HTTP Server running at http://0.0.0.0:{PORT}")
            httpd.serve_forever()
    
    http_thread = threading.Thread(target=start_http, daemon=True)
    http_thread.start()
    
    # Start WebSocket server and keep it running
    async with websockets.serve(server.handle_client, "0.0.0.0", 8765):
        print("✅ WebSocket server is running and waiting for connections...")
        print("💡 Press Ctrl+C to stop the server")
        try:
            # Keep the server running indefinitely
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    asyncio.run(main())
