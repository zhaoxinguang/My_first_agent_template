"""EmbeddingClient：默认配置、provider 校验、健康探测不抛异常。"""

import pytest

from memory.embedding import EmbeddingClient


def test_ollama_defaults(monkeypatch):
    for name in (
        "EMBEDDING_PROVIDER",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    client = EmbeddingClient()

    assert client.provider == "ollama"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "nomic-embed-text"


def test_rejects_unknown_provider():
    with pytest.raises(ValueError, match="不支持的 embedding provider"):
        EmbeddingClient(provider="cohere")


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        EmbeddingClient(provider="openai")


def test_openai_provider_uses_default_model(monkeypatch):
    client = EmbeddingClient(
        provider="openai", api_key="test-key", model="text-embedding-3-small"
    )

    assert client.provider == "openai"
    assert client.model == "text-embedding-3-small"


def test_health_check_reports_unreachable_service():
    """指向一个没人监听的端口：应返回 (False, 原因)，且不抛异常。"""
    client = EmbeddingClient(base_url="http://127.0.0.1:65535/v1")

    ok, detail = client.health_check(timeout=1.0)

    assert ok is False
    assert detail


def test_health_check_never_raises_on_broken_client():
    """即使底层客户端对象异常，也只能是 False，不能让启动流程崩。"""
    client = EmbeddingClient()
    client.client = object()  # 没有 models 属性

    ok, detail = client.health_check()

    assert ok is False
    assert "AttributeError" in detail


def test_embed_query_rejects_blank_text():
    client = EmbeddingClient()

    with pytest.raises(ValueError, match="不能为空"):
        client.embed_query("   ")


def test_embed_documents_rejects_blank_items():
    client = EmbeddingClient()

    with pytest.raises(ValueError, match="空字符串"):
        client.embed_documents(["ok", ""])


def test_embed_documents_empty_list():
    client = EmbeddingClient()

    assert client.embed_documents([]) == []
