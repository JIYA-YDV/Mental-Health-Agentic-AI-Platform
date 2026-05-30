"""
Configuration Package

Centralized settings management using pydantic-settings.
All environment variables and application configuration
are loaded and validated here.

Usage:
    from backend.config.settings import settings
    print(settings.APP_NAME)
    print(settings.EMOTION_MODEL)
"""

from backend.config.settings import settings

__all__ = ["settings"]