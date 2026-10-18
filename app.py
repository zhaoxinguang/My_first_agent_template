"""应用入口：组装记忆、agent 和界面。

启动：
    python app.py

启动前需要：
1. 复制 .env 并填好模型配置（或本地先起 ollama）；
2. 可选：启动 embedding 服务（缺省时自动退化为无向量检索）。
"""

from agent.factory import create_agent
from Gradio_UI import GradioUI
from memory import (
    DEFAULT_USER_ID,
    EmbeddingClient,
    MemoryAgent,
    MemoryRetriever,
    MemoryStore,
)


def create_embedding_client():
    """Embedding 属于增强能力，失败时降级而不是让程序起不来。

    注意区分两件事：
    - 构造失败（配置错误，例如 openai provider 缺 key）→ 返回 None；
    - 服务不可用（ollama 没起、模型没 pull）→ 客户端仍返回，
      让检索层回退到结构化偏好，并在这里把真实状态打印出来，
      避免「看起来有记忆、实际一条没注入」。
    """
    try:
        client = EmbeddingClient()
    except Exception as exc:
        print(f"[memory] Embedding 初始化失败，将退化为普通偏好读取：{exc}")
        return None

    ok, detail = client.health_check()
    if ok:
        print(
            f"[memory] Embedding 服务可用："
            f"{client.provider} / {client.model}"
        )
    else:
        print(
            f"[memory] Embedding 服务不可用（{detail}）。\n"
            f"[memory] 偏好检索将回退到结构化读取；"
            f"恢复向量检索请先在本地启动并拉取模型：\n"
            f"    ollama serve\n"
            f"    ollama pull {client.model}"
        )
    return client


def main():
    # 1. 记忆层：一个 store 贯穿存储与检索
    store = MemoryStore()
    retriever = MemoryRetriever(
        store=store,
        embedding_client=create_embedding_client(),
    )

    # 2. 执行层：底层 agent + 记忆注入包装
    agent = MemoryAgent(
        agent=create_agent(),
        retriever=retriever,
        user_id=DEFAULT_USER_ID,
    )

    # 3. 展示层：UI 只负责对话和 run 记录，不重复注入偏好
    GradioUI(
        agent,
        memory_store=store,
        user_id=DEFAULT_USER_ID,
    ).launch()


if __name__ == "__main__":
    main()
