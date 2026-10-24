"""应用入口：组装记忆、agent 和界面。

启动：
    python app.py

启动前需要：
1. 复制 .env 并填好模型配置（或本地先起 ollama）；
2. 可选：启动 embedding 服务（缺省时自动退化为无向量检索）。
"""

from agent.factory import create_agent
from agent.model import create_model
from Gradio_UI import GradioUI
from config.settings import ACTIVE_MODEL, MODEL_PROFILES
from memory import (
    DEFAULT_USER_ID,
    EmbeddingClient,
    HistoryIndex,
    MemoryAgent,
    MemoryRetriever,
    MemoryStore,
    PreferenceManager,
    TurnDistiller,
)


def log_active_model():
    """启动时打印实际生效的模型，避免「改了 .env 却不知道有没有生效」。"""
    profile = MODEL_PROFILES.get(ACTIVE_MODEL, {})
    print(
        f"[model] 生效档位：ACTIVE_MODEL={ACTIVE_MODEL} "
        f"→ model_id={profile.get('model_id')} "
        f"@ {profile.get('api_base')}"
    )


def probe_model_endpoint():
    """探一下模型接口是否连通（只测网络，不校验密钥和模型名）。

    只用于排障：任何 HTTP 状态码都算连通；连不上或超时才提示。
    探活失败不阻止启动——有些环境要等代理起来才有网。
    """
    import httpx

    api_base = MODEL_PROFILES.get(ACTIVE_MODEL, {}).get("api_base")
    if not api_base:
        return

    try:
        response = httpx.get(api_base, timeout=3.0)
    except Exception as exc:
        print(
            f"[probe] {api_base} 不可达：{type(exc).__name__}: {exc}\n"
            f"[probe] 本地档位先确认 ollama 正在跑（ollama serve / ollama list）；"
            f"云端档位检查网络或代理。"
        )
        return

    print(f"[probe] {api_base} 可达（HTTP {response.status_code}）")


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


def backfill_history_index(history_index):
    """启动时给「还没有当前模型向量」的历史任务补建向量。

    只补缺失的，所以正常情况下是 0 条；首次升级、或换过 embedding 模型时
    会把已有历史一次性索引好——否则这些老对话永远不会出现在召回结果里。
    """
    if history_index.embedding_client is None:
        print(
            "[memory] 未启用 embedding：历史对话只存文本，"
            "不参与语义召回（偏好仍走结构化读取）"
        )
        return

    try:
        stats = history_index.backfill()
    except Exception as exc:  # 索引失败不该阻止启动
        print(
            "[memory] 历史向量补建失败，跳过："
            f"{type(exc).__name__}: {exc}"
        )
        return

    if stats["total"] == 0:
        print(f"[memory] 历史向量已是最新（模型 {stats['model']}）")
        return

    print(
        f"[memory] 历史向量补建完成：{stats['success']}/{stats['total']} 条"
        f"（模型 {stats['model']}，失败 {stats['failed']}）"
    )


def log_preference_budget(preference_manager, retriever=None):
    """启动时打印偏好预算占用。

    偏好现在是「目录注入 + 按需加载」，条目数直接影响每轮固定开销，
    所以这里提前把占用暴露出来——预算满了会在写入时被拒，
    与其跑到对话里才发现，不如启动就说清楚。

    ``retriever`` 只为多打一句「目录实际显示几条」：预算按库里真实行数算，
    但同一维度可能躺着好几条同义记录（简短 / short / low），目录里只显示一条。
    不写清楚的话，用户看到「5 条」对照目录里的 2 行，会以为记忆没生效。
    """
    try:
        status = preference_manager.budget_status(DEFAULT_USER_ID)
    except Exception as exc:  # 只读日志，失败不阻止启动
        print(f"[memory] 偏好预算统计失败：{type(exc).__name__}: {exc}")
        return

    print(
        f"[memory] 用户偏好：{status['items']} 条 / "
        f"{status['used_chars']} 字符"
        f"（上限 {status['max_items']} 条 / {status['char_limit']} 字符）"
    )
    if retriever is not None:
        try:
            shown = len(
                retriever.retrieve_preference_index(
                    user_id=DEFAULT_USER_ID
                )
            )
        except Exception as exc:
            print(f"[memory] 目录条数统计失败：{type(exc).__name__}: {exc}")
        else:
            if shown and shown < status["items"]:
                print(
                    f"[memory] 目录按维度合并后实际显示 {shown} 条"
                    "（相同维度的旧记录不再重复列出）"
                )
    if not status["ok"]:
        print(
            "[memory] 偏好已超出预算：新偏好会被拒绝写入，"
            "请用 forget_user_preference 清理或用 set_preference.py 合并条目"
        )


def log_distiller(distiller):
    """启动时说明「每轮自动提炼」是否开启。

    这件事直接影响两件事：用户偏好会不会自己长出来、
    每轮会不会多一次模型调用（本地 ollama 下能感知到）。
    与其让人猜，不如启动时打清楚。
    """
    if not distiller.enabled:
        print(
            "[memory] 自动提炼：已关闭（PREF_AUTO_EXTRACT=0）。"
            "偏好只能靠模型主动调 save_user_preference 写入"
        )
        return

    print(
        "[memory] 自动提炼：开启（每轮结束后台提炼 0~"
        f"{distiller.max_preferences} 条偏好 + 本轮摘要）"
    )


def main():
    # 0. 先把实际生效的模型打出来：换模型排障时第一眼看这里
    log_active_model()
    probe_model_endpoint()

    # 1. 记忆层：一个 store 贯穿存储与检索
    store = MemoryStore()
    embedding_client = create_embedding_client()
    retriever = MemoryRetriever(
        store=store,
        embedding_client=embedding_client,
    )

    # 1.1 偏好写侧：动态更新的入口（agent 工具与 set_preference.py 共用）
    preference_manager = PreferenceManager(
        store=store,
        embedding_client=embedding_client,
    )
    log_preference_budget(preference_manager, retriever)

    # 1.2 每轮结束的自动提炼：偏好 + 本轮摘要，运行时保证，不靠模型自觉
    distiller = TurnDistiller(
        store=store,
        embedding_client=embedding_client,
        model_factory=create_model,
    )
    log_distiller(distiller)

    # 1.3 历史对话索引：同一套 embedding（默认 ollama 本地模型），
    #     启动时补齐缺失向量，之后每轮结束由 GradioUI 增量写入。
    history_index = HistoryIndex(
        store=store,
        embedding_client=embedding_client,
    )
    backfill_history_index(history_index)

    # 2. 执行层：底层 agent + 记忆注入包装。
    #    工具与注入必须共用同一个 store：模型刚 save 的偏好，
    #    下一轮目录才能立刻读到（动态更新的关键）。
    agent = MemoryAgent(
        agent=create_agent(
            store=store,
            embedding_client=embedding_client,
        ),
        retriever=retriever,
        user_id=DEFAULT_USER_ID,
    )

    # 3. 展示层：UI 只负责对话和 run 记录，不重复注入偏好
    GradioUI(
        agent,
        memory_store=store,
        user_id=DEFAULT_USER_ID,
        history_index=history_index,
        distiller=distiller,
    ).launch()


if __name__ == "__main__":
    main()
