#!/usr/bin/env python3
"""
Simple TTS from PKL File
Interactive terminal-based TTS using stored voice profile (saves to files)
"""

import os
import sys
import torch
import torchaudio
import pickle
from fireredtts2.fireredtts2 import FireRedTTS2

class SimpleTTS:
    def __init__(self, voice_profile_path):
        self.voice_profile_path = voice_profile_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.voice_profile = None
        self.output_counter = 0
        
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
    
    def generate_and_save(self, text):
        """Generate audio and save to file"""
        print(f"🎤 Generating audio for: {text}")
        
        try:
            # Generate audio
            audio = self.model.generate_monologue(
                text=text,
                prompt_wav=self.voice_profile['original_audio_path'],
                prompt_text=self.voice_profile['transcription'],
                temperature=0.8,
                topk=30
            )
            
            # Save to file
            self.output_counter += 1
            output_file = f"ali_tts_output_{self.output_counter}.wav"
            torchaudio.save(output_file, audio.cpu(), 24000)
            
            print(f"✅ Audio generated and saved: {output_file}")
            print(f"📊 Audio shape: {audio.shape}")
            print(f"⏱️  Duration: {audio.shape[1] / 24000:.2f} seconds")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return None

def main():
    """Main interactive loop"""
    print("🎭 Simple TTS with Ali's Voice (File Output)")
    print("=" * 60)
    
    # Check if voice profile exists
    voice_profile_path = "voice_profiles/ali_voice_profile.pkl"
    if not os.path.exists(voice_profile_path):
        print(f"❌ Voice profile not found: {voice_profile_path}")
        print("Please run create_ali_voice_pkl.py first!")
        sys.exit(1)
    
    # Initialize TTS system
    tts = SimpleTTS(voice_profile_path)
    
    print("\n🎤 TTS is ready!")
    print("📝 Type your text and press Enter to generate speech")
    print("💡 Audio files will be saved as ali_tts_output_*.wav")
    print("💡 Type 'quit', 'exit', or 'q' to stop")
    print("💡 Type 'status' to see system status")
    print("💡 Type 'list' to see generated files")
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
                print(f"   - Generated files: {tts.output_counter}")
                print(f"   - Voice profile: {tts.voice_profile_path}")
                continue
            elif text.lower() == 'list':
                print(f"📁 Generated audio files:")
                for i in range(1, tts.output_counter + 1):
                    filename = f"ali_tts_output_{i}.wav"
                    if os.path.exists(filename):
                        size = os.path.getsize(filename) / 1024
                        print(f"   - {filename} ({size:.1f} KB)")
                continue
            
            # Generate and save audio
            output_file = tts.generate_and_save(text)
            
            if output_file:
                print(f"🎵 You can play the file: {output_file}")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()



