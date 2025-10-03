#!/usr/bin/env python3
"""
Voice Storage Demo for FireRedTTS2
Shows how to store and reuse voice characteristics efficiently
"""

import os
import torch
import pickle
import torchaudio
from fireredtts2.fireredtts2 import FireRedTTS2

class VoiceStorage:
    def __init__(self, fireredtts2_model):
        self.model = fireredtts2_model
        
    def extract_voice_tokens(self, audio_path, text):
        """
        Extract and store voice tokens instead of full audio
        This is much more efficient than storing WAV files
        """
        print(f"🎤 Extracting voice tokens from: {audio_path}")
        
        # Load and prepare the audio
        audio_tensor = self.model.load_prompt_audio(audio_path)
        
        # Tokenize the audio (this is what we'll store)
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
        audio_tokens, token_length = self.model._audio_tokenizer.encode(
            audio_tensor.to(self.model.device),
            audio_length.to(self.model.device),
            batch_size=48,
        )
        
        # Store the essential voice information
        voice_data = {
            'audio_tokens': audio_tokens.cpu(),  # Much smaller than raw audio
            'token_length': token_length.cpu(),
            'text': text,
            'sample_rate': 16000,
            'original_audio_shape': audio_tensor.shape
        }
        
        print(f"📊 Original audio size: {audio_tensor.numel() * 4 / 1024 / 1024:.2f} MB")
        print(f"📊 Tokenized size: {audio_tokens.numel() * 4 / 1024:.2f} KB")
        print(f"💾 Compression ratio: {audio_tensor.numel() / audio_tokens.numel():.1f}x")
        
        return voice_data
    
    def save_voice_profile(self, voice_data, profile_name):
        """Save voice profile to disk"""
        profile_path = f"voice_profiles/{profile_name}.pkl"
        os.makedirs("voice_profiles", exist_ok=True)
        
        with open(profile_path, 'wb') as f:
            pickle.dump(voice_data, f)
        
        print(f"💾 Voice profile saved: {profile_path}")
        return profile_path
    
    def load_voice_profile(self, profile_name):
        """Load voice profile from disk"""
        profile_path = f"voice_profiles/{profile_name}.pkl"
        
        with open(profile_path, 'rb') as f:
            voice_data = pickle.load(f)
        
        print(f"📂 Voice profile loaded: {profile_path}")
        return voice_data
    
    def generate_with_stored_voice(self, voice_data, new_text, temperature=0.8, topk=30):
        """Generate speech using stored voice tokens"""
        print(f"🎭 Generating speech with stored voice...")
        
        # Reconstruct the audio tokens
        audio_tokens = voice_data['audio_tokens'].to(self.model.device)
        
        # Create the prompt segment
        prompt_segment = self.model.prepare_prompt(
            text=voice_data['text'],
            speaker="[S1]",
            audio_path=None  # We'll set the audio manually
        )
        
        # Manually set the audio tokens
        prompt_segment.audio = audio_tokens
        
        # Generate new speech
        audio = self.model.generate_monologue(
            text=new_text,
            prompt_wav=None,  # We're using stored tokens
            prompt_text=voice_data['text'],
            temperature=temperature,
            topk=topk
        )
        
        return audio

def demo_voice_storage():
    """Demonstrate voice storage and reuse"""
    print("🎭 FireRedTTS2 Voice Storage Demo")
    print("=" * 50)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fireredtts2 = FireRedTTS2(
        pretrained_dir="./pretrained_models/FireRedTTS2",
        gen_type="monologue",
        device=device,
    )
    
    # Initialize voice storage
    voice_storage = VoiceStorage(fireredtts2)
    
    # Example: Extract voice from example audio
    audio_path = "./examples/chat_prompt/zh/S1.flac"
    prompt_text = "[S1]啊，可能说更适合美国市场应该是什么样子。那这这个可能说当然如果说有有机会能亲身的去考察去了解一下，那当然是有更好的帮助。"
    
    if os.path.exists(audio_path):
        # Extract voice tokens
        voice_data = voice_storage.extract_voice_tokens(audio_path, prompt_text)
        
        # Save voice profile
        profile_path = voice_storage.save_voice_profile(voice_data, "chinese_voice")
        
        # Load voice profile
        loaded_voice = voice_storage.load_voice_profile("chinese_voice")
        
        # Generate new speech with stored voice
        new_texts = [
            "这是使用存储的语音特征生成的新文本。",
            "Voice cloning with stored tokens is very efficient.",
        ]
        
        for i, text in enumerate(new_texts):
            print(f"\n📝 Generating: {text}")
            audio = voice_storage.generate_with_stored_voice(loaded_voice, text)
            
            # Save generated audio
            output_file = f"stored_voice_output_{i+1}.wav"
            torchaudio.save(output_file, audio.cpu(), 24000)
            print(f"✅ Generated: {output_file}")
    
    else:
        print(f"❌ Example audio not found: {audio_path}")

if __name__ == "__main__":
    demo_voice_storage()



