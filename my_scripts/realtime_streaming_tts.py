#!/usr/bin/env python3
"""
Real-time Streaming TTS with Ali's Voice
Interactive terminal-based TTS using stored voice profile
"""

import os
import sys
import torch
import torchaudio
import pickle
import threading
import queue
import time
import pyaudio
import wave
import numpy as np
from fireredtts2.fireredtts2 import FireRedTTS2

class RealtimeTTS:
    def __init__(self, voice_profile_path):
        self.voice_profile_path = voice_profile_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.voice_profile = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.is_generating = False
        
        # Audio settings
        self.sample_rate = 24000
        self.chunk_size = 1024
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Load voice profile and model
        self.load_voice_profile()
        self.load_model()
        
    def load_voice_profile(self):
        """Load the voice profile from PKL file"""
        print("🔄 Loading voice profile...")
        try:
            with open(self.voice_profile_path, 'rb') as f:
                self.voice_profile = pickle.load(f)
            
            print("✅ Voice profile loaded successfully!")
            print(f"📊 Profile info:")
            print(f"   - Original audio: {self.voice_profile['original_audio_path']}")
            print(f"   - Transcription: {self.voice_profile['transcription']}")
            print(f"   - Duration: {self.voice_profile['extraction_info']['duration_seconds']:.2f} seconds")
            print(f"   - Compression ratio: {self.voice_profile['extraction_info']['compression_ratio']:.1f}x")
            
        except Exception as e:
            print(f"❌ Error loading voice profile: {e}")
            sys.exit(1)
    
    def load_model(self):
        """Load the FireRedTTS2 model"""
        print("🔄 Loading FireRedTTS2 model...")
        try:
            self.model = FireRedTTS2(
                pretrained_dir=self.voice_profile['model_info']['pretrained_dir'],
                gen_type=self.voice_profile['model_info']['gen_type'],
                device=self.device,
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def generate_audio_chunk(self, text, chunk_size=50):
        """Generate audio in chunks for streaming"""
        try:
            # Generate full audio
            audio = self.model.generate_monologue(
                text=text,
                prompt_wav=self.voice_profile['original_audio_path'],
                prompt_text=self.voice_profile['transcription'],
                temperature=0.8,
                topk=30
            )
            
            # Convert to numpy array
            audio_np = audio.squeeze().cpu().numpy()
            
            # Split into chunks for streaming
            chunk_length = int(self.sample_rate * chunk_size / 1000)  # chunk_size in ms
            chunks = []
            
            for i in range(0, len(audio_np), chunk_length):
                chunk = audio_np[i:i + chunk_length]
                if len(chunk) > 0:
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return []
    
    def audio_player_thread(self):
        """Audio playback thread"""
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        
        while True:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                if audio_chunk is None:  # Shutdown signal
                    break
                
                # Play the chunk
                stream.write(audio_chunk.tobytes())
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
    
    def generate_and_play(self, text):
        """Generate audio and play it in real-time"""
        if self.is_generating:
            print("⏳ Already generating audio, please wait...")
            return
        
        self.is_generating = True
        print(f"🎤 Generating audio for: {text}")
        
        try:
            # Generate audio chunks
            audio_chunks = self.generate_audio_chunk(text, chunk_size=100)  # 100ms chunks
            
            if not audio_chunks:
                print("❌ Failed to generate audio")
                return
            
            print(f"🎵 Playing {len(audio_chunks)} audio chunks...")
            
            # Queue audio chunks for playback
            for i, chunk in enumerate(audio_chunks):
                self.audio_queue.put(chunk.astype(np.float32))
                if i == 0:
                    print("🔊 Audio started playing...")
            
            print("✅ Audio generation completed!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.is_generating = False
    
    def start_audio_thread(self):
        """Start the audio playback thread"""
        self.audio_thread = threading.Thread(target=self.audio_player_thread, daemon=True)
        self.audio_thread.start()
        print("🔊 Audio playback thread started")
    
    def cleanup(self):
        """Cleanup resources"""
        self.audio_queue.put(None)  # Signal shutdown
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        self.p.terminate()
        print("🧹 Cleanup completed")

def main():
    """Main interactive loop"""
    print("🎭 Real-time Streaming TTS with Ali's Voice")
    print("=" * 60)
    
    # Check if voice profile exists
    voice_profile_path = "voice_profiles/ali_voice_profile.pkl"
    if not os.path.exists(voice_profile_path):
        print(f"❌ Voice profile not found: {voice_profile_path}")
        print("Please run create_ali_voice_pkl.py first!")
        sys.exit(1)
    
    # Initialize TTS system
    tts = RealtimeTTS(voice_profile_path)
    tts.start_audio_thread()
    
    print("\n🎤 Real-time TTS is ready!")
    print("📝 Type your text and press Enter to generate speech")
    print("💡 Type 'quit', 'exit', or 'q' to stop")
    print("💡 Type 'status' to see system status")
    print("-" * 60)
    
    try:
        while True:
            # Get user input
            text = input("\n🎤 Enter text: ").strip()
            
            if not text:
                continue
            
            # Handle special commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif text.lower() == 'status':
                print(f"📊 System Status:")
                print(f"   - Device: {tts.device}")
                print(f"   - Generating: {tts.is_generating}")
                print(f"   - Queue size: {tts.audio_queue.qsize()}")
                continue
            
            # Generate and play audio
            tts.generate_and_play(text)
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        tts.cleanup()

if __name__ == "__main__":
    main()



