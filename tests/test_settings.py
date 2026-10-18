"""config.settings：执行预算可被环境变量覆盖，非法值要显式报错。"""

import importlib

import pytest

import config.settings as settings

ENV_KEYS = (
    "AGENT_MAX_STEPS",
    "AGENT_MAX_TOKENS",
    "AGENT_TEMPERATURE",
    "AGENT_VERBOSITY_LEVEL",
)


@pytest.fixture
def reload_settings(monkeypatch):
    """每个用例结束后把模块恢复成默认配置，避免污染其它测试。"""
    yield lambda: importlib.reload(settings)

    for name in ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(settings)


def test_defaults_when_env_missing(monkeypatch, reload_settings):
    for name in ENV_KEYS:
        monkeypatch.delenv(name, raising=False)

    module = reload_settings()

    assert module.MAX_STEPS == 6
    assert module.MAX_TOKENS == 2048
    assert module.TEMPERATURE == 0.5
    assert module.VERBOSITY_LEVEL == 2


def test_env_overrides(monkeypatch, reload_settings):
    monkeypatch.setenv("AGENT_MAX_STEPS", "12")
    monkeypatch.setenv("AGENT_MAX_TOKENS", "4096")
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.1")
    monkeypatch.setenv("AGENT_VERBOSITY_LEVEL", "0")

    module = reload_settings()

    assert module.MAX_STEPS == 12
    assert module.MAX_TOKENS == 4096
    assert module.TEMPERATURE == 0.1
    assert module.VERBOSITY_LEVEL == 0


def test_blank_env_falls_back_to_default(monkeypatch, reload_settings):
    monkeypatch.setenv("AGENT_MAX_STEPS", "  ")

    assert reload_settings().MAX_STEPS == 6


def test_invalid_int_raises(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "abc")

    with pytest.raises(ValueError, match="AGENT_MAX_STEPS"):
        importlib.reload(settings)

    monkeypatch.delenv("AGENT_MAX_STEPS", raising=False)
    importlib.reload(settings)  # 恢复模块可用状态


def test_invalid_float_raises(monkeypatch):
    monkeypatch.setenv("AGENT_TEMPERATURE", "hot")

    with pytest.raises(ValueError, match="AGENT_TEMPERATURE"):
        importlib.reload(settings)

    monkeypatch.delenv("AGENT_TEMPERATURE", raising=False)
    importlib.reload(settings)


def test_prompt_path_points_to_existing_file():
    assert settings.PROMPT_PATH.is_file()


def test_model_profiles_cover_active_model():
    assert settings.ACTIVE_MODEL in settings.MODEL_PROFILES
    assert settings.MODEL_PROFILES[settings.ACTIVE_MODEL]["model_id"]
