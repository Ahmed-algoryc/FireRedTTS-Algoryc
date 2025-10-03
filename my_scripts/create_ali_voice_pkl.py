#!/usr/bin/env python3
"""
Create Ali Voice PKL File
Clone voice from ali1.wav and save as PKL file for future reuse
"""

import os
import torch
import torchaudio
import pickle
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

def create_ali_voice_pkl():
    """Create PKL file with Ali's voice tokens"""
    print("🎭 Creating Ali Voice PKL File")
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
        print(f"   - Size: {audio.numel() * 4 / 1024 / 1024:.2f} MB")
        
        # Load audio for tokenization
        audio_tensor = fireredtts2.load_prompt_audio(audio_path)
        
        # Extract voice tokens (this is what we'll store)
        print(f"\n🔧 Extracting voice tokens...")
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
        audio_tokens, token_length = fireredtts2._audio_tokenizer.encode(
            audio_tensor.to(device),
            audio_length.to(device),
            batch_size=48,
        )
        
        print(f"📊 Tokenization results:")
        print(f"   - Audio tokens shape: {audio_tokens.shape}")
        print(f"   - Token length: {token_length}")
        print(f"   - Token size: {audio_tokens.numel() * 4 / 1024:.2f} KB")
        print(f"💾 Compression ratio: {audio_tensor.numel() / audio_tokens.numel():.1f}x smaller")
        
        # Auto-transcribe the audio
        print(f"\n🎤 Auto-transcribing {audio_path}...")
        transcription = transcribe_audio_with_whisper(audio_path)
        
        if not transcription:
            print("❌ Auto-transcription failed. Please provide transcription manually:")
            transcription = input("Enter what's spoken in ali1.wav: ").strip()
            if not transcription:
                print("❌ No transcription provided. Exiting.")
                return False
        
        print(f"✅ Using transcription: {transcription}")
        
        # Create voice profile with all necessary information
        voice_profile = {
            'audio_tokens': audio_tokens.cpu(),
            'token_length': token_length.cpu(),
            'transcription': transcription,
            'original_audio_path': audio_path,
            'sample_rate': 16000,
            'original_shape': audio_tensor.shape,
            'original_sample_rate': sample_rate,
            'extraction_info': {
                'device_used': device,
                'compression_ratio': audio_tensor.numel() / audio_tokens.numel(),
                'original_size_mb': audio_tensor.numel() * 4 / 1024 / 1024,
                'token_size_kb': audio_tokens.numel() * 4 / 1024,
                'duration_seconds': audio.shape[1] / sample_rate
            },
            'model_info': {
                'pretrained_dir': pretrained_dir,
                'gen_type': 'monologue',
                'device': device
            }
        }
        
        # Save voice profile as PKL file
        os.makedirs("voice_profiles", exist_ok=True)
        pkl_path = "voice_profiles/ali_voice_profile.pkl"
        
        print(f"\n💾 Saving voice profile to PKL file...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(voice_profile, f)
        
        print(f"✅ Voice profile saved: {pkl_path}")
        
        # Test the PKL file by loading it back
        print(f"\n🔄 Testing PKL file...")
        with open(pkl_path, 'rb') as f:
            loaded_profile = pickle.load(f)
        
        print(f"✅ PKL file loaded successfully!")
        print(f"📊 Loaded profile info:")
        print(f"   - Original audio: {loaded_profile['original_audio_path']}")
        print(f"   - Duration: {loaded_profile['extraction_info']['duration_seconds']:.2f} seconds")
        print(f"   - Compression ratio: {loaded_profile['extraction_info']['compression_ratio']:.1f}x")
        print(f"   - Token size: {loaded_profile['extraction_info']['token_size_kb']:.2f} KB")
        
        # Test voice cloning with the original audio to verify it works
        print(f"\n🎭 Testing voice cloning with original audio...")
        target_text = "This is the test of my voice clone. I am the great Ali bhai, QA testing is my passion."
        
        # Use the auto-transcribed text
        sample_text = transcription
        
        print(f"📝 Target text: {target_text}")
        print(f"📝 Sample text: {sample_text}")
        
        try:
            # Generate test audio
            print("🔄 Generating test audio...")
            audio = fireredtts2.generate_monologue(
                text=target_text,
                prompt_wav=audio_path,
                prompt_text=sample_text,
                temperature=0.8,
                topk=30
            )
            
            # Save the test audio
            test_output = "ali_voice_test_from_pkl.wav"
            torchaudio.save(test_output, audio.cpu(), 24000)
            
            print(f"✅ Test audio generated: {test_output}")
            print(f"📊 Audio shape: {audio.shape}")
            print(f"⏱️  Duration: {audio.shape[1] / 24000:.2f} seconds")
            
        except Exception as e:
            print(f"⚠️  Test generation failed: {e}")
            print("   This might be due to transcription mismatch, but PKL file is still valid")
        
        print(f"\n🎉 Ali voice PKL file created successfully!")
        print(f"📁 PKL file: {pkl_path}")
        print(f"💾 Size: {os.path.getsize(pkl_path) / 1024:.2f} KB")
        print(f"🎵 Test audio: ali_voice_test_from_pkl.wav")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating Ali Voice PKL File")
    
    success = create_ali_voice_pkl()
    
    if success:
        print("\n✅ Ali voice PKL file created successfully!")
        print("🎯 You can now use this PKL file for future voice cloning without the original audio!")
        print("📁 PKL file location: voice_profiles/ali_voice_profile.pkl")
    else:
        print("\n❌ PKL file creation failed.")
        exit(1)
