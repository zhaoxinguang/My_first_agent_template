"""测试共享夹具：每个用例用独立的临时 SQLite 库，互不污染。"""

import pytest

from memory.store import MemoryStore
from tests.helpers import FakeEmbeddingClient


@pytest.fixture
def store(tmp_path):
    """指向 tmp_path 的 MemoryStore，测试结束随 tmp_path 一起丢弃。"""
    return MemoryStore(db_path=tmp_path / "memory.db")


@pytest.fixture
def fake_embedding():
    """返回 FakeEmbeddingClient 工厂，方便按用例定制行为。"""
    return FakeEmbeddingClient
