"""UI 开关：share/debug 必须默认关闭，且可用环境变量打开。

默认 share=True 会把界面直接开到公网，这是之前硬编码在 launch() 里的行为，
现在已经改成「默认关闭 + 显式开启」，这里把这条约束固定住。
"""

import inspect

from Gradio_UI import GradioUI, _env_flag


def test_env_flag_defaults_to_false(monkeypatch):
    monkeypatch.delenv("GRADIO_SHARE", raising=False)

    assert _env_flag("GRADIO_SHARE") is False


def test_env_flag_accepts_truthy_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("GRADIO_SHARE", value)
        assert _env_flag("GRADIO_SHARE") is True


def test_env_flag_treats_other_values_as_false(monkeypatch):
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("GRADIO_SHARE", value)
        assert _env_flag("GRADIO_SHARE") is False


def test_env_flag_falls_back_to_given_default(monkeypatch):
    monkeypatch.delenv("GRADIO_DEBUG", raising=False)

    assert _env_flag("GRADIO_DEBUG", True) is True


def test_launch_defaults_to_none_so_env_decides():
    parameters = inspect.signature(GradioUI.launch).parameters

    assert parameters["share"].default is None
    assert parameters["debug"].default is None
