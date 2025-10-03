#!/usr/bin/env python3
"""
Auto Transcribe and Clone Voice
Automatically transcribes ali1.wav and then generates voice clone
"""

import os
import torch
import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2

def transcribe_audio_with_whisper(audio_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        import whisper
        print("🔄 Loading Whisper model for transcription...")
        
        # Load Whisper model (base model is good balance of speed/accuracy)
        model = whisper.load_model("base")
        
        print(f"🎤 Transcribing {audio_path}...")
        result = model.transcribe(audio_path)
        
        transcription = result["text"].strip()
        print(f"✅ Transcription: {transcription}")
        
        return transcription
        
    except ImportError:
        print("❌ Whisper not installed. Installing...")
        os.system("pip install openai-whisper")
        return transcribe_audio_with_whisper(audio_path)
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def transcribe_audio_with_speechrecognition(audio_path):
    """Alternative transcription using speech_recognition library"""
    try:
        import speech_recognition as sr
        print("🔄 Using speech_recognition for transcription...")
        
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        
        # Transcribe
        print(f"🎤 Transcribing {audio_path}...")
        transcription = r.recognize_google(audio)
        
        print(f"✅ Transcription: {transcription}")
        return transcription
        
    except ImportError:
        print("❌ speech_recognition not installed. Installing...")
        os.system("pip install SpeechRecognition pydub")
        return transcribe_audio_with_speechrecognition(audio_path)
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def auto_transcribe_and_clone():
    """Automatically transcribe and clone voice"""
    print("🎭 Auto Transcribe and Voice Clone")
    print("=" * 50)
    
    # Check if ali1.wav exists
    audio_path = "ali1.wav"
    if not os.path.exists(audio_path):
        print(f"❌ Error: {audio_path} not found!")
        return False
    
    print(f"✅ Found audio file: {audio_path}")
    
    # Try to transcribe the audio
    print("\n🎤 Attempting automatic transcription...")
    
    # Try Whisper first (more accurate)
    transcription = transcribe_audio_with_whisper(audio_path)
    
    # If Whisper fails, try speech_recognition
    if not transcription:
        print("\n🔄 Trying alternative transcription method...")
        transcription = transcribe_audio_with_speechrecognition(audio_path)
    
    # If both fail, ask user to provide transcription
    if not transcription:
        print("\n❌ Automatic transcription failed.")
        print("Please provide the transcription manually:")
        transcription = input("Enter what's spoken in ali1.wav: ").strip()
        
        if not transcription:
            print("❌ No transcription provided. Exiting.")
            return False
    
    print(f"\n✅ Using transcription: {transcription}")
    
    # Check if models exist
    pretrained_dir = "./pretrained_models/FireRedTTS2"
    if not os.path.exists(pretrained_dir):
        print(f"❌ Error: Pretrained models not found at {pretrained_dir}")
        return False
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Initialize the model
        print("🔄 Loading FireRedTTS2 model...")
        fireredtts2 = FireRedTTS2(
            pretrained_dir=pretrained_dir,
            gen_type="monologue",
            device=device,
        )
        print("✅ Model loaded successfully!")
        
        # The text you want to generate
        target_text = "This is the test of my voice clone. I am the great Ali bhai, QA testing is my passion."
        
        print(f"\n🎭 Generating voice clone...")
        print(f"📝 Target text: {target_text}")
        print(f"📝 Sample text (auto-transcribed): {transcription}")
        
        # Generate the voice clone
        print("🔄 Generating audio...")
        audio = fireredtts2.generate_monologue(
            text=target_text,
            prompt_wav=audio_path,
            prompt_text=transcription,
            temperature=0.8,
            topk=30
        )
        
        # Save the generated audio
        output_file = "ali_voice_clone_auto.wav"
        torchaudio.save(output_file, audio.cpu(), 24000)
        
        print(f"✅ Voice clone generated successfully!")
        print(f"🎵 Output file: {output_file}")
        print(f"📊 Audio shape: {audio.shape}")
        print(f"⏱️  Duration: {audio.shape[1] / 24000:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Auto Transcribe and Voice Clone")
    
    success = auto_transcribe_and_clone()
    
    if success:
        print("\n✅ Voice cloning completed successfully!")
        print("🎵 Check the generated file: ali_voice_clone_auto.wav")
    else:
        print("\n❌ Voice cloning failed.")
        exit(1)
