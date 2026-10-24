from typing import Any

from smolagents import OpenAIServerModel

from config.settings import (
    ACTIVE_MODEL,
    MODEL_PROFILES,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    REASONING_EFFORT,
    VALID_REASONING_EFFORTS,
)


def _thinking_kwargs(provider: str, effort: str, supports_thinking: bool = True) -> dict[str, Any]:
    """将思考配置转换为 API 请求参数。

    Args:
        provider: 档位名（ollama / llama / deepseek）。
        effort: 目标思考强度；空字符串表示「不传」。
        supports_thinking: 该档的模型是否支持思考。为 False 时一律不发——
            Ollama 对不支持思考的模型（如 llama3.2）会给带 reasoning_effort
            的请求直接返回 400，而不是文档说的「无害忽略」。
    """
    if not effort or not supports_thinking:
        return {}

    if provider == "deepseek":
        return {} if effort == "none" else {"reasoning_effort": effort}

    return {"reasoning_effort": effort}


class ThinkingAwareModel(OpenAIServerModel):
    """按服务端能力发送思考字段，并在被拒绝时自动降级。

    为什么需要这一层：思考开关的字段名和「不传会怎样」各家不同，
    而且 Ollama 的判定粒度是**具体模型**而不是服务端版本。用户在 ``.env``
    里换 ``OLLAMA_MODEL_ID``（比如换成 gemma3）时，档位声明的能力就过期了。
    与其让用户记住每种模型的脾气，不如在第一次被拒时自己摘掉这个字段：
    400 的错误信息里明确带 ``does not support thinking``，可以安全地据此降级，
    并把结论记在实例上，避免每次请求都先失败再重试。
    """

    #: Ollama 拒绝思考字段时返回的错误片段
    UNSUPPORTED_MARK = "does not support thinking"

    def __init__(self, *args, thinking_kwargs=None, **kwargs):
        # 先记下「本该发什么」，再交给父类塞进 self.kwargs
        self._thinking_kwargs = dict(thinking_kwargs or {})
        self.thinking_rejected = False
        super().__init__(*args, **{**kwargs, **self._thinking_kwargs})

    def _sync_thinking_kwargs(self):
        """把 kwargs 里的思考字段调整到当前该有的状态。"""
        for key in self._thinking_kwargs:
            if self.thinking_rejected or not self._thinking_kwargs:
                self.kwargs.pop(key, None)
            else:
                self.kwargs[key] = self._thinking_kwargs[key]

    def __call__(self, messages, **kwargs):
        self._sync_thinking_kwargs()
        try:
            return super().__call__(messages, **kwargs)
        except Exception as exc:
            if not self._thinking_kwargs or self.thinking_rejected:
                raise
            if self.UNSUPPORTED_MARK not in str(exc):
                raise
            # 模型不支持思考：摘掉字段重试一次，并记住别再发
            self.thinking_rejected = True
            self._sync_thinking_kwargs()
            print(
                "[model] 该模型不支持思考，已自动关闭 reasoning_effort："
                f"{self.model_id}"
            )
            return super().__call__(messages, **kwargs)


def create_model(reasoning_effort: str | None = None) -> OpenAIServerModel:
    """创建模型实例，可指定思考强度。"""
    if ACTIVE_MODEL not in MODEL_PROFILES:
        raise ValueError(
            f"未知模型配置：{ACTIVE_MODEL}，"
            f"可选值：{list(MODEL_PROFILES)}"
        )

    config = MODEL_PROFILES[ACTIVE_MODEL]

    if ACTIVE_MODEL == "deepseek" and not config["api_key"]:
        raise ValueError("请在 .env 中配置 DEEPSEEK_API_KEY")

    effort = REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    effort = (effort or "").strip().lower()

    if effort and effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort 取值非法：{effort!r}，"
            f"可选值：{list(VALID_REASONING_EFFORTS)}，或留空表示不传"
        )

    extra_kwargs: dict[str, Any] = _thinking_kwargs(
        ACTIVE_MODEL,
        effort,
        supports_thinking=config.get("supports_thinking", True),
    )

    return ThinkingAwareModel(
        model_id=config["model_id"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        thinking_kwargs=extra_kwargs,
        client_kwargs={
            "timeout": REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES,
        },
        **extra_kwargs,
    )

