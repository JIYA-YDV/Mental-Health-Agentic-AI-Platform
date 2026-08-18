# -*- coding: utf-8 -*-
"""
LLM Response Generator using Groq API.
Generates empathetic streaming responses based on emotion analysis.
"""

import os
import sys
from pathlib import Path
from typing import Generator, Optional

# ============================================================
# PATH SETUP + ENV LOADING (must run BEFORE any imports that need env)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env explicitly from project root
from dotenv import load_dotenv
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

from groq import Groq

try:
    from backend.config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    settings = None


# ============================================================
# CONSTANTS
# ============================================================

# Default model - Groq's free compound systems model (replaces versatile)
DEFAULT_MODEL = "groq/compound-mini"

CRISIS_REDIRECT_MESSAGE = (
    "I can see you might be going through an extremely difficult time. "
    "Please know that immediate support is available:\n\n"
    "**Crisis Helpline: 988** (call or text, 24/7)\n"
    "**Crisis Text Line: Text HOME to 741741**\n"
    "**Emergency Services: 911**\n\n"
    "You don't have to face this alone. Please reach out to these "
    "trained professionals right now."
)

SYSTEM_PROMPT = (
    "You are a compassionate mental health support assistant. Your role is to:\n\n"
    "1. Acknowledge the user's emotions with genuine empathy\n"
    "2. Validate their feelings without judgment\n"
    "3. Offer ONE practical coping strategy relevant to their emotion\n"
    "4. Encourage professional help when appropriate\n"
    "5. Keep responses warm, human, and concise (3-4 sentences max)\n\n"
    "CRITICAL RULES:\n"
    "- NEVER provide medical diagnoses\n"
    "- NEVER claim to replace professional therapy\n"
    "- NEVER dismiss or minimize feelings\n"
    "- If crisis keywords detected: STOP and redirect to crisis resources ONLY\n"
    "- Always end with encouragement or a gentle question\n\n"
    "Tone: Warm, calm, non-clinical, like a caring friend who understands mental health."
)


class LLMResponder:
    """Generates empathetic streaming responses using Groq's LLM API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Priority: passed arg > env var > settings > default
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or (getattr(settings, "GROQ_API_KEY", None) if SETTINGS_AVAILABLE else None)
        )

        self.model = (
            model
            or os.getenv("GROQ_MODEL")
            or (getattr(settings, "GROQ_MODEL", None) if SETTINGS_AVAILABLE else None)
            or DEFAULT_MODEL
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Get a free key at https://console.groq.com and add "
                "GROQ_API_KEY=your_key to the .env file in project root."
            )

        self.client = Groq(api_key=self.api_key)

    def build_prompt(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        crisis_detected: bool = False,
        safety_override_applied: bool = False,
    ) -> Optional[str]:
        """Build a context-aware prompt for the LLM."""
        if crisis_detected:
            return None

        override_note = ""
        if safety_override_applied:
            override_note = "Note: Our safety system flagged this input as potentially high-risk."

        prompt = (
            f'The user shared: "{user_text}"\n\n'
            f"Our emotion analysis detected: {emotion} "
            f"(confidence: {confidence:.0%})\n"
            f"{override_note}\n\n"
            f"Please respond with warmth and empathy. Acknowledge their {emotion}, "
            f"validate what they're feeling, and offer one gentle, practical suggestion. "
            f'Keep it to 3-4 sentences. Start directly with empathy - no "I understand that..."'
        )
        return prompt

    def stream_response(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        crisis_detected: bool = False,
        safety_override_applied: bool = False,
    ) -> Generator[str, None, None]:
        """Stream an empathetic LLM response chunk by chunk."""

        # Crisis override - never call LLM
        if crisis_detected:
            yield CRISIS_REDIRECT_MESSAGE
            return

        prompt = self.build_prompt(
            user_text, emotion, confidence,
            crisis_detected, safety_override_applied,
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
                stream=True,
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

        except Exception as e:
            # Graceful fallback - never crash the UI
            yield f"\n\nI hear you. What you're feeling - {emotion} - is valid. "
            yield "Please consider reaching out to a mental health professional "
            yield "for personalized support."
            print(f"[LLM Error] {e}", file=sys.stderr)

    def get_full_response(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        crisis_detected: bool = False,
        safety_override_applied: bool = False,
    ) -> str:
        """Non-streaming version - returns complete response at once."""
        if crisis_detected:
            return CRISIS_REDIRECT_MESSAGE

        chunks = list(self.stream_response(
            user_text, emotion, confidence,
            crisis_detected, safety_override_applied,
        ))
        return "".join(chunks)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Responder")
    print("=" * 60)

    # Debug info
    print(f"\n[DEBUG] Project root: {PROJECT_ROOT}")
    print(f"[DEBUG] .env path:    {ENV_PATH}")
    print(f"[DEBUG] .env exists:  {ENV_PATH.exists()}")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key and SETTINGS_AVAILABLE:
        api_key = getattr(settings, "GROQ_API_KEY", "") or ""

    if not api_key:
        print("\n[ERROR] GROQ_API_KEY not found!")
        print("\nTo fix:")
        print("  1. Get free key at: https://console.groq.com")
        print(f"  2. Create/edit file: {ENV_PATH}")
        print("  3. Add this line:  GROQ_API_KEY=gsk_your_key_here")
        print("     (no spaces, no quotes)")
        sys.exit(1)

    print(f"\n[OK] API key found (length: {len(api_key)}, starts with: {api_key[:6]}...)")

    responder = LLMResponder()
    print(f"[OK] Model: {responder.model}")

    # Test 1: Normal sad input
    print("\n" + "=" * 60)
    print("Test 1: Sadness input (should stream response)")
    print("=" * 60)
    print("\nStreaming response:\n")
    for chunk in responder.stream_response(
        user_text="I've been feeling really sad and empty lately",
        emotion="sadness",
        confidence=0.85,
        crisis_detected=False,
    ):
        print(chunk, end="", flush=True)

    # Test 2: Crisis input
    print("\n\n" + "=" * 60)
    print("Test 2: Crisis input (should show static message)")
    print("=" * 60)
    print("\nStreaming response:\n")
    for chunk in responder.stream_response(
        user_text="I don't want to be here anymore",
        emotion="sadness",
        confidence=0.9,
        crisis_detected=True,
    ):
        print(chunk, end="", flush=True)

    print("\n\n[OK] All tests complete!")