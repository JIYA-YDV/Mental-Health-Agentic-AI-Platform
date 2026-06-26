# tests/unit/test_settings.py
"""Test centralized configuration validation."""
import pytest
from pydantic import ValidationError


pytestmark = pytest.mark.unit


def test_settings_loads_with_defaults():
    """Settings should instantiate with no env vars set."""
    from backend.config.settings import Settings
    s = Settings()
    assert s.APP_NAME == "Mental Health Agentic AI Platform"
    assert s.PORT == 8000
    assert 0.0 <= s.SIMILARITY_THRESHOLD <= 1.0


def test_settings_singleton_is_not_a_tuple():
    """Regression test for the trailing-comma bug we fixed."""
    from backend.config.settings import settings
    assert not isinstance(settings, tuple), \
        "settings singleton must NOT be a tuple (trailing-comma bug)"
    assert hasattr(settings, "APP_VERSION")


def test_log_level_validator_accepts_valid_levels():
    from backend.config.settings import Settings
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        s = Settings(LOG_LEVEL=level)
        assert s.LOG_LEVEL == level


def test_log_level_validator_rejects_invalid():
    from backend.config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(LOG_LEVEL="VERBOSE")


def test_log_level_validator_normalizes_to_upper():
    from backend.config.settings import Settings
    s = Settings(LOG_LEVEL="info")
    assert s.LOG_LEVEL == "INFO"


def test_port_range_validation():
    from backend.config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(PORT=70000)
    with pytest.raises(ValidationError):
        Settings(PORT=0)


def test_similarity_threshold_range():
    from backend.config.settings import Settings
    with pytest.raises(ValidationError):
        Settings(SIMILARITY_THRESHOLD=1.5)
    with pytest.raises(ValidationError):
        Settings(SIMILARITY_THRESHOLD=-0.1)


def test_explainer_settings_present():
    """Verify the new explainer settings exist after refactor."""
    from backend.config.settings import settings
    assert hasattr(settings, "EXPLAINER_DISPLAY_THRESHOLD")
    assert hasattr(settings, "EXPLAINER_MIN_TOKENS")
    assert hasattr(settings, "EXPLAINER_MAX_TOKENS")
    assert hasattr(settings, "FALLBACK_THRESHOLD_DELTA")