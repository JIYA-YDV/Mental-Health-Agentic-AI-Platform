"""
LLM Response Generator using Groq API.
Generates empathetic, streaming responses based on emotion analysis.
"""

import os
from typing import Generator, Optional
from groq import Groq
from backend.config.settings import settings

# Safety: topics we NEVER comment on beyond crisis resources
CRISIS_REDIRECT_MESSAGE = """
I can see you might be going through an extremely difficult time. 
Please know that immediate support is available:

🆘 **Crisis Helpline: 988** (call or text, 24/7)
🆘 **Crisis Text Line: Text HOME to 741741**
🆘 **Emergency Services: 911**

You don't have to face this alone. Please reach out to these 
trained professionals right now. 💙
"""

SYSTEM_PROMPT = """
You are a compassionate mental health support assistant. Your role is to:

1. Acknowledge the user's emotions with genuine empathy
2. Validate their feelings without judgment  
3. Offer ONE practical coping strategy relevant to their emotion
4. Encourage professional help when appropriate
5. Keep responses warm, human, and concise (3-4 sentences max)

CRITICAL RULES:
- NEVER provide medical diagnoses
- NEVER claim to replace professional therapy
- NEVER dismiss or minimize feelings
- If crisis keywords detected: STOP and redirect to crisis resources ONLY
- Always end with encouragement or a gentle question

Tone: Warm, calm, non-clinical, like a caring friend who understands mental health.
"""


class LLMResponder:
    """Generates empathetic streaming responses using Groq's LLM API."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", None)
        
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        
        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b-8192"  # Fast, free, good quality
    
    def build_prompt(
        self, 
        user_text: str, 
        emotion: str,
        confidence: float,
        crisis_detected: bool,
        safety_override_applied: bool
    ) -> Optional[str]:
        """
        Build a context-aware prompt using analysis results.
        The LLM gets richer context than just the raw text.
        """
        # If crisis detected — don't generate creative response
        if crisis_detected:
            return None  # Signal to use static crisis message
        
        prompt = f"""
The user wrote: "{user_text}"

Our emotion analysis detected: {emotion} (confidence: {confidence:.0%})
{"Note: Safety system flagged this as high-risk content." if safety_override_applied else ""}

Please provide a compassionate, brief response that:
1. Acknowledges their {emotion}
2. Validates what they're feeling
3. Offers one gentle, practical suggestion
4. Feels human and warm, not clinical

Keep it to 3-4 sentences. Start directly with empathy, not with "I understand that..."
"""
        return prompt
    
    def stream_response(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        crisis_detected: bool = False,
        safety_override_applied: bool = False
    ) -> Generator[str, None, None]:
        """
        Stream an empathetic LLM response word by word.
        
        Yields: individual text chunks as they arrive from Groq
        """
        # Crisis override — never generate AI response for crisis
        if crisis_detected:
            yield CRISIS_REDIRECT_MESSAGE
            return
        
        prompt = self.build_prompt(
            user_text, emotion, confidence, 
            crisis_detected, safety_override_applied
        )
        
        try:
            # Groq streaming API call
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,       # Some creativity, not too random
                max_tokens=200,        # Keep responses concise
                stream=True            # ← THE KEY: enables streaming
            )
            
            # Yield each chunk as it arrives
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
                    
        except Exception:
            # Graceful fallback — never crash the UI
            yield f"\n\nI'm here to support you. "
            yield f"What you're feeling — {emotion} — is valid and real. "
            yield "Please consider reaching out to a mental health professional "
            yield "who can provide personalized support. 💙"
    
    def get_full_response(
        self,
        user_text: str,
        emotion: str, 
        confidence: float,
        crisis_detected: bool = False,
        safety_override_applied: bool = False
    ) -> str:
        """Non-streaming version — returns complete response at once."""
        if crisis_detected:
            return CRISIS_REDIRECT_MESSAGE
        
        chunks = list(self.stream_response(
            user_text, emotion, confidence,
            crisis_detected, safety_override_applied
        ))
        return "".join(chunks)


# Test it standalone:
if __name__ == "__main__":
    responder = LLMResponder()
    
    print("Testing streaming response...\n")
    print("Response: ", end="")
    
    for chunk in responder.stream_response(
        user_text="I've been feeling really sad and empty lately",
        emotion="sadness",
        confidence=0.85,
        crisis_detected=False
    ):
        print(chunk, end="", flush=True)
    
    print("\n\nDone!")