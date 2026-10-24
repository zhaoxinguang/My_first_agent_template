"""思考（思维链）开关：默认开启，且按服务端差异翻译成正确的请求字段。

背景：Ollama 的 OpenAI 兼容端点对支持思考的模型**默认开思考**，
请求里不带任何思考字段就会一直烧思维链；而它的原生 ``think: true/false``
在这个端点上被静默忽略。所以开关只能靠 ``reasoning_effort``：
ollama 上 ``none`` 关闭、``high`` 开启；DeepSeek 恰好相反——
不传就是不思考，且枚举里没有 ``none``（传了会 400）。

默认值取 ``high``（开）：实测在 agent 场景下 qwen3.5:9b 通常不产生
思考内容，开着几乎免费；需要「绝对不想」时再设 none。
"""

import importlib
from types import SimpleNamespace

import pytest

import agent.model as agent_model
import config.settings as settings

PROFILES = {
    "ollama": {
        "model_id": "qwen3.5:9b",
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
        "supports_thinking": True,
    },
    "llama": {
        "model_id": "llama3.2:latest",
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
        # llama3.2 不支持思考；发 reasoning_effort 会被 Ollama 判 400
        "supports_thinking": False,
    },
    "deepseek": {
        "model_id": "deepseek-flash",
        "api_base": "https://api.deepseek.com",
        "api_key": "sk-test",
        "supports_thinking": True,
    },
}


@pytest.fixture
def ollama(monkeypatch):
    monkeypatch.setattr(agent_model, "ACTIVE_MODEL", "ollama")
    monkeypatch.setattr(agent_model, "MODEL_PROFILES", PROFILES)
    return agent_model


@pytest.fixture
def deepseek(monkeypatch):
    monkeypatch.setattr(agent_model, "ACTIVE_MODEL", "deepseek")
    monkeypatch.setattr(agent_model, "MODEL_PROFILES", PROFILES)
    return agent_model


@pytest.fixture
def llama(monkeypatch):
    monkeypatch.setattr(agent_model, "ACTIVE_MODEL", "llama")
    monkeypatch.setattr(agent_model, "MODEL_PROFILES", PROFILES)
    return agent_model


def _body(model):
    """取出真正会进请求体的参数（不含 messages）。"""
    kwargs = model._prepare_completion_kwargs(
        messages=[{"role": "user", "content": "hi"}]
    )
    return {k: v for k, v in kwargs.items() if k != "messages"}


# ---------- ollama：不传就默认开，必须显式 none ----------


def test_default_effort_disables_thinking(ollama, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")

    assert _body(ollama.create_model())["reasoning_effort"] == "none"


def test_explicit_effort_is_forwarded(ollama, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")

    assert _body(ollama.create_model("high"))["reasoning_effort"] == "high"


def test_blank_effort_sends_nothing(ollama, monkeypatch):
    """留空＝交给服务端默认（ollama 上等于开思考），属于显式选择。"""
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "")

    assert "reasoning_effort" not in _body(ollama.create_model())


def test_effort_is_case_and_space_insensitive(ollama, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", " none ")

    assert _body(ollama.create_model())["reasoning_effort"] == "none"


# ---------- deepseek：不传就是不思考，传 none 会 400 ----------


def test_deepseek_omits_field_when_disabled(deepseek, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")

    assert "reasoning_effort" not in _body(deepseek.create_model())


def test_deepseek_forwards_supported_level(deepseek, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")

    assert _body(deepseek.create_model("medium"))["reasoning_effort"] == "medium"


# ---------- 非法值要显式报错 ----------


def test_invalid_effort_raises(ollama, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")

    with pytest.raises(ValueError, match="reasoning_effort"):
        ollama.create_model("thinking-a-lot")


def test_ollama_branch_knobs_untouched(ollama, monkeypatch):
    """别把既有参数覆盖掉：max_tokens / temperature 必须还在请求体里。"""
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")
    body = _body(ollama.create_model())

    assert body["max_tokens"] == agent_model.MAX_TOKENS
    assert body["temperature"] == agent_model.TEMPERATURE


# ---------- llama 档：同一个 ollama 服务，只是换个模型名 ----------


def test_llama_profile_is_selectable(llama, monkeypatch):
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "high")
    model = llama.create_model()

    assert model.model_id == "llama3.2:latest"
    # api_base 落在底层 client 上（OpenAIServerModel 不直接暴露该属性）
    assert str(model.client.base_url).startswith("http://localhost:11434/v1")


def test_llama_does_not_send_thinking_field(llama, monkeypatch):
    """llama3.2 不支持思考：Ollama 会直接 400，必须一个字段都不发。"""
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "high")

    assert "reasoning_effort" not in _body(llama.create_model())


def test_profile_without_thinking_support_never_sends(llama, monkeypatch):
    """即使显式要求 none，不支持思考的档也不发字段。"""
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "none")

    assert "reasoning_effort" not in _body(llama.create_model("none"))


# ---------- 换模型后档位能力会过期：被拒时自动降级 ----------


class _Boom(Exception):
    pass


class _FakeClient:
    """记录每次请求体；前 N 次抛「不支持思考」，之后正常返回。"""

    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.bodies = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.bodies.append(dict(kwargs))
                if len(outer.bodies) <= outer.fail_times:
                    raise _Boom(
                        '"qwen3.5:9b" does not support thinking'
                    )

                class _Msg:
                    role, content, tool_calls = "assistant", "ok", None

                    def model_dump(self, include=None):
                        return {
                            key: getattr(self, key)
                            for key in (include or ("role", "content"))
                        }

                return SimpleNamespace(
                    choices=[SimpleNamespace(message=_Msg())],
                    usage=SimpleNamespace(
                        prompt_tokens=1, completion_tokens=1, total_tokens=2
                    ),
                )

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _model_with_fake_client(monkeypatch, thinking_kwargs, fail_times=1):
    monkeypatch.setattr(agent_model, "ACTIVE_MODEL", "ollama")
    monkeypatch.setattr(agent_model, "MODEL_PROFILES", PROFILES)
    monkeypatch.setattr(agent_model, "REASONING_EFFORT", "high")
    model = agent_model.create_model()
    fake = _FakeClient(fail_times=fail_times)
    monkeypatch.setattr(model, "client", fake)
    return model, fake


def test_model_drops_thinking_field_when_server_rejects_it(monkeypatch):
    model, fake = _model_with_fake_client(monkeypatch, {"reasoning_effort": "high"})

    model([{"role": "user", "content": "hi"}])

    # 第一次带字段被拒，第二次去掉后成功
    assert len(fake.bodies) == 2
    assert fake.bodies[0].get("reasoning_effort") == "high"
    assert "reasoning_effort" not in fake.bodies[1]
    assert model.thinking_rejected is True


def test_rejection_is_remembered_across_calls(monkeypatch):
    model, fake = _model_with_fake_client(monkeypatch, {"reasoning_effort": "high"})

    model([{"role": "user", "content": "hi"}])
    model([{"role": "user", "content": "again"}])

    # 第 3 次调用直接不发字段，不再先失败一次
    assert "reasoning_effort" not in fake.bodies[2]
    assert len(fake.bodies) == 3


def test_unrelated_error_still_raises(monkeypatch):
    """只对「不支持思考」这一个错误降级，别的错误必须照常抛出。"""
    model, fake = _model_with_fake_client(monkeypatch, {"reasoning_effort": "high"})

    class _OtherError(_Boom):
        def __str__(self):
            return "context length exceeded"

    monkeypatch.setattr(
        fake.chat.completions, "create",
        lambda **kw: (_ for _ in ()).throw(_OtherError()),
    )

    with pytest.raises(_OtherError):
        model([{"role": "user", "content": "hi"}])


def test_llama_profile_exists_in_settings(monkeypatch):
    """档位表里必须有 llama，否则 .env 里写 ACTIVE_MODEL=llama 会启动即报错。"""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: True)
    monkeypatch.delenv("LLAMA_MODEL_ID", raising=False)
    module = importlib.reload(settings)

    try:
        assert module.MODEL_PROFILES["llama"]["model_id"] == "llama3.2:latest"
        # 默认复用 ollama 的地址：换模型不该要求再填一遍 host
        assert (
            module.MODEL_PROFILES["llama"]["api_base"]
            == module.MODEL_PROFILES["ollama"]["api_base"]
        )
    finally:
        importlib.reload(settings)


# ---------- 后台提炼固定不思考 ----------


def test_distiller_always_disables_thinking(monkeypatch, tmp_path):
    """提炼只要 JSON：不该跟随 AGENT_REASONING_EFFORT 一起变慢。"""
    from memory.distiller import TurnDistiller
    from memory.store import MemoryStore

    calls = []

    def fake_create_model(reasoning_effort=None):
        calls.append(reasoning_effort)
        return object()

    monkeypatch.setattr("agent.model.create_model", fake_create_model, raising=True)

    distiller = TurnDistiller(store=MemoryStore(db_path=tmp_path / "m.db"))
    distiller._get_model()

    assert calls == ["none"]


# ---------- settings 层的取值校验 ----------


@pytest.fixture
def reload_settings(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: True)
    yield lambda: importlib.reload(settings)
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)
    importlib.reload(settings)


def test_settings_default_enables_thinking(reload_settings, monkeypatch):
    """默认开：agent 场景下模型多半不思考，开着是免费的；
    真开始思考时才需要用户主动改成 none。"""
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    assert reload_settings().REASONING_EFFORT == "high"


def test_settings_rejects_unknown_level(reload_settings, monkeypatch):
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "lots")

    with pytest.raises(ValueError, match="AGENT_REASONING_EFFORT"):
        importlib.reload(settings)


def test_settings_allows_blank(reload_settings, monkeypatch):
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "  ")

    assert reload_settings().REASONING_EFFORT == ""
