import os
import base64
import requests
import json

class VoiceService:
    def __init__(self):
        """Initialize ElevenLabs with API key from environment."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")
        
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def generate_speech(self, text: str, output_format="mp3") -> bytes:
        """
        Convert text to speech using ElevenLabs API directly.
        
        Args:
            text (str): Text to convert to speech
            output_format (str): Audio format (mp3, wav, etc.)
            
        Returns:
            bytes: Audio data as bytes
        """
        try:
            # Truncate text if too long (ElevenLabs has character limits)
            if len(text) > 2500:
                text = text[:2500] + "..."
            
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"ElevenLabs API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def generate_speech_base64(self, text: str) -> str:
        """
        Generate speech and return as base64 string for web playback.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            str: Base64 encoded audio data
        """
        audio_bytes = self.generate_speech(text)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None
    
    def test_voice(self, test_text="Hello! I'm Kryptonic AI, your crypto companion."):
        """Test the voice service with sample text."""
        try:
            print("Testing ElevenLabs API connection...")
            print(f"API Key present: {'Yes' if self.api_key else 'No'}")
            print(f"Voice ID: {self.voice_id}")
            
            audio_bytes = self.generate_speech(test_text)
            if audio_bytes:
                # Save test audio file
                with open("test_voice.mp3", "wb") as f:
                    f.write(audio_bytes)
                print("✅ Voice test successful! Check 'test_voice.mp3' file.")
                print(f"Audio size: {len(audio_bytes)} bytes")
                return True
            else:
                print("❌ Voice test failed!")
                return False
        except Exception as e:
            print(f"❌ Voice test error: {e}")
            return False
    
    def get_available_voices(self):
        """Get list of available voices from ElevenLabs."""
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                voices = response.json()
                print("Available voices:")
                for voice in voices.get('voices', [])[:5]:  # Show first 5 voices
                    print(f"- {voice['name']}: {voice['voice_id']}")
                return voices
            else:
                print(f"Error fetching voices: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error fetching voices: {e}")
            return None