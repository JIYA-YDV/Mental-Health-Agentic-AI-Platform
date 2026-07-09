# tests/unit/test_llm_responder.py

import pytest
from unittest.mock import MagicMock, patch

class TestLLMResponder:
    
    def test_crisis_returns_static_message(self):
        """Crisis input should never use LLM — return static resources."""
        from backend.models.llm_responder import LLMResponder, CRISIS_REDIRECT_MESSAGE
        
        with patch("backend.models.llm_responder.Groq"):
            responder = LLMResponder.__new__(LLMResponder)
            responder.client = MagicMock()
            
            chunks = list(responder.stream_response(
                user_text="I want to end my life",
                emotion="sadness",
                confidence=0.9,
                crisis_detected=True  # ← crisis flag
            ))
            
            full_response = "".join(chunks)
            assert "988" in full_response  # crisis number present
            assert "Crisis" in full_response
    
    def test_normal_input_calls_groq(self):
        """Non-crisis input should call Groq API."""
        from backend.models.llm_responder import LLMResponder
        
        with patch("backend.models.llm_responder.Groq") as MockGroq:
            # Mock the streaming response
            mock_chunk = MagicMock()
            mock_chunk.choices[0].delta.content = "I hear you."
            
            mock_stream = [mock_chunk]
            MockGroq.return_value.chat.completions.create.return_value = mock_stream
            
            responder = LLMResponder()
            chunks = list(responder.stream_response(
                user_text="I feel sad",
                emotion="sadness",
                confidence=0.8,
                crisis_detected=False
            ))
            
            assert len(chunks) > 0
            MockGroq.return_value.chat.completions.create.assert_called_once()
    
    def test_build_prompt_includes_emotion(self):
        """Prompt should include detected emotion for context."""
        from backend.models.llm_responder import LLMResponder
        
        with patch("backend.models.llm_responder.Groq"):
            responder = LLMResponder.__new__(LLMResponder)
            responder.client = MagicMock()
            
            prompt = responder.build_prompt(
                user_text="I feel anxious",
                emotion="fear",
                confidence=0.78,
                crisis_detected=False,
                safety_override_applied=False
            )
            
            assert "fear" in prompt.lower()
            assert "anxious" in prompt.lower()
    
    def test_api_error_returns_fallback(self):
        """API failure should return graceful fallback, not crash."""
        from backend.models.llm_responder import LLMResponder
        
        with patch("backend.models.llm_responder.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = Exception("API Error")
            
            responder = LLMResponder()
            
            # Should not raise — should yield fallback message
            chunks = list(responder.stream_response(
                user_text="I feel overwhelmed",
                emotion="fear",
                confidence=0.7,
                crisis_detected=False
            ))
            
            full = "".join(chunks)
            assert len(full) > 0  # got something
            # No crash = test passes