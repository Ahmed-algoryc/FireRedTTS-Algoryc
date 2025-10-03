#!/usr/bin/env python3
"""
Generate Ali's Voice Clone
Simple script to clone voice from ali1.wav and generate specific text
"""

import os
import torch
import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2

def generate_ali_voice_clone():
    """Generate voice clone from ali1.wav"""
    print("🎭 Ali Voice Clone Generation")
    print("=" * 50)
    
    # Check if ali1.wav exists
    audio_path = "ali1.wav"
    if not os.path.exists(audio_path):
        print(f"❌ Error: {audio_path} not found!")
        return False
    
    print(f"✅ Found audio file: {audio_path}")
    
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
        
        # Load and analyze the audio file
        print(f"\n🎤 Analyzing {audio_path}...")
        audio, sample_rate = torchaudio.load(audio_path)
        print(f"📊 Audio info:")
        print(f"   - Shape: {audio.shape}")
        print(f"   - Sample rate: {sample_rate} Hz")
        print(f"   - Duration: {audio.shape[1] / sample_rate:.2f} seconds")
        
        # The text you want to generate
        target_text = "This is the test of my voice clone. I am the great Ali bhai, QA testing is my passion."
        
        # IMPORTANT: You need to provide the transcription of what's actually said in ali1.wav
        # Please replace this with the actual text that's spoken in your ali1.wav file
        sample_text = "Please provide the actual transcription of ali1.wav here"
        
        print(f"⚠️  WARNING: You need to provide the actual transcription of ali1.wav")
        print(f"   Current sample text: {sample_text}")
        print(f"   Please edit the script and replace 'sample_text' with what's actually spoken in ali1.wav")
        
        print(f"\n🎭 Generating voice clone...")
        print(f"📝 Target text: {target_text}")
        print(f"📝 Sample text: {sample_text}")
        
        # Generate the voice clone
        print("🔄 Generating audio...")
        audio = fireredtts2.generate_monologue(
            text=target_text,
            prompt_wav=audio_path,
            prompt_text=sample_text,
            temperature=0.8,
            topk=30
        )
        
        # Save the generated audio
        output_file = "ali_voice_clone.wav"
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
    print("🚀 Ali Voice Clone Generation")
    
    success = generate_ali_voice_clone()
    
    if success:
        print("\n✅ Voice cloning completed successfully!")
        print("🎵 Check the generated file: ali_voice_clone.wav")
    else:
        print("\n❌ Voice cloning failed.")
        exit(1)
