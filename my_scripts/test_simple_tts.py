#!/usr/bin/env python3
"""
Simple TTS test script for FireRedTTS2
Tests basic TTS functionality with random voices
"""

import os
import sys
import torch
import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2

def test_simple_tts():
    """Test simple TTS with random voices"""
    print("🔥 FireRedTTS2 Simple TTS Test")
    print("=" * 50)
    
    # Check if models exist
    pretrained_dir = "./pretrained_models/FireRedTTS2"
    if not os.path.exists(pretrained_dir):
        print(f"❌ Error: Pretrained models not found at {pretrained_dir}")
        return False
    
    print(f"✅ Found pretrained models at {pretrained_dir}")
    
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
            gen_type="monologue",  # Start with monologue for simplicity
            device=device,
        )
        print("✅ Model loaded successfully!")
        
        # Test texts in different languages
        test_texts = [
            "Hello everyone, welcome to our newly launched FireRedTTS2. It supports multiple languages including English, Chinese, Japanese, Korean, French, German, and Russian.",
            "如果你厌倦了千篇一律的AI音色，不满意于其他模型语言支持不够丰富，那么本项目将会成为你绝佳的工具。",
            "ランダムな話者と言語を選択して合成できます",
            "J'évolue constamment et j'espère pouvoir parler davantage de langues avec plus d'aisance à l'avenir.",
        ]
        
        print("\n🎤 Testing TTS generation...")
        for i, text in enumerate(test_texts):
            print(f"\n📝 Text {i+1}: {text[:50]}...")
            
            try:
                # Generate audio with random voice
                audio = fireredtts2.generate_monologue(
                    text=text,
                    temperature=0.8,
                    topk=30
                )
                
# Save the audio
import os
os.makedirs('tests/output', exist_ok=True)
output_file = f"tests/output/test_output_{i+1}.wav"
torchaudio.save(output_file, audio.cpu(), 24000)
                print(f"✅ Generated audio saved as: {output_file}")
                print(f"📊 Audio shape: {audio.shape}")
                
            except Exception as e:
                print(f"❌ Error generating audio for text {i+1}: {e}")
                continue
        
        print("\n🎉 TTS test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_tts()
    if success:
        print("\n✅ All tests passed! FireRedTTS2 is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)
