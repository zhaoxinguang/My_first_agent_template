"""MemoryRetriever：向量召回、模型过滤、两条降级路径、历史任务召回。"""

import pytest

from memory.retriever import (
    PREF_INDEX_MAX_ITEMS,
    PREF_INDEX_VALUE_CHARS,
    MemoryRetriever,
)
from tests.helpers import FakeEmbeddingClient, FakeSemanticEmbeddingClient


def _seed(store):
    store.save_preference("u", "response_style", "detail", "回答简洁分步骤")
    store.save_preference("u", "format", "language", "中文")


# ---------- 余弦相似度 ----------

def test_cosine_similarity_identical_vectors():
    # 浮点累计误差，用 approx 而不是精确相等
    assert MemoryRetriever._cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert MemoryRetriever._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_guards_dimension_mismatch():
    assert MemoryRetriever._cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_cosine_similarity_guards_zero_norm():
    assert MemoryRetriever._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_guards_empty():
    assert MemoryRetriever._cosine_similarity([], []) == 0.0


# ---------- 向量召回 ----------

def test_retrieve_preferences_uses_vectors_when_available(store):
    _seed(store)
    client = FakeEmbeddingClient()
    # 先按同一模型写入向量
    from memory.preference_manager import PreferenceManager

    PreferenceManager(store=store, embedding_client=client).save_preference(
        "u", "format", "language", "中文"
    )

    retriever = MemoryRetriever(store=store, embedding_client=client)
    results = retriever.retrieve_preferences("中文", user_id="u")

    assert len(results) == 1
    assert "similarity" in results[0]
    assert client.calls  # 确实调用了 embedding 服务


def test_retrieve_preferences_respects_limit(store):
    for index in range(5):
        store.save_preference("u", "cat", f"key{index}", f"value{index}")

    retriever = MemoryRetriever(store=store, embedding_client=None)
    assert len(retriever.retrieve_preferences("任意", user_id="u", limit=2)) == 2


# ---------- 降级路径 ----------

def test_falls_back_to_structured_without_client(store):
    _seed(store)

    retriever = MemoryRetriever(store=store, embedding_client=None)
    results = retriever.retrieve_preferences("任意", user_id="u")

    assert {r["memory_key"] for r in results} == {"detail", "language"}
    assert all("similarity" not in r for r in results)


def test_falls_back_when_embedding_service_fails(store):
    """ollama 没起时必须回退，而不是整轮失忆。"""
    _seed(store)
    retriever = MemoryRetriever(
        store=store, embedding_client=FakeEmbeddingClient(fail=True)
    )

    results = retriever.retrieve_preferences("任意", user_id="u")

    assert {r["memory_key"] for r in results} == {"detail", "language"}


def test_falls_back_when_model_mismatch(store):
    """换过 embedding 模型后旧向量被过滤，此时应回退而不是返回空。"""
    from memory.preference_manager import PreferenceManager

    PreferenceManager(
        store=store, embedding_client=FakeEmbeddingClient(model="old-model")
    ).save_preference("u", "format", "language", "中文")

    retriever = MemoryRetriever(
        store=store, embedding_client=FakeEmbeddingClient(model="new-model")
    )
    results = retriever.retrieve_preferences("中文", user_id="u")

    assert [r["memory_key"] for r in results] == ["language"]


def test_vectors_from_other_model_are_not_mixed(store):
    """存在当前模型向量时，不能把别的模型的向量算进去。"""
    from memory.preference_manager import PreferenceManager

    PreferenceManager(
        store=store, embedding_client=FakeEmbeddingClient(model="old-model")
    ).save_preference("u", "cat", "old", "旧模型向量")
    client = FakeEmbeddingClient(model="new-model")
    PreferenceManager(store=store, embedding_client=client).save_preference(
        "u", "cat", "new", "新模型向量"
    )

    retriever = MemoryRetriever(store=store, embedding_client=client)
    results = retriever.retrieve_preferences("任意", user_id="u")

    assert [r["memory_key"] for r in results] == ["new"]


def test_empty_query_returns_empty(store):
    _seed(store)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert retriever.retrieve_preferences("   ", user_id="u") == []
    assert retriever.retrieve_preferences("任意", user_id="u", limit=0) == []


# ---------- 格式化 ----------

def test_format_preferences_empty():
    assert "没有检索到" in MemoryRetriever.format_preferences([])


def test_format_preferences_lists_each_item(store):
    _seed(store)
    retriever = MemoryRetriever(store=store, embedding_client=None)
    text = retriever.format_preferences(
        retriever.retrieve_preferences("任意", user_id="u")
    )

    assert "language" in text
    assert "中文" in text
    assert text.count("- 类别：") == 2


# ---------- 历史任务召回 ----------

def test_retrieve_history_prefers_success(store):
    failed = store.start_run("查天气")
    store.finish_run(failed, final_answer="失败了", status="failed")
    ok = store.start_run("查天气")
    store.finish_run(ok, final_answer="晴", status="success")

    retriever = MemoryRetriever(store=store, embedding_client=None)
    results = retriever.retrieve("查天气")

    assert results[0]["status"] == "success"
    assert results[0]["run_id"] == ok


def test_retrieve_history_empty_keyword(store):
    retriever = MemoryRetriever(store=store, embedding_client=None)
    assert retriever.retrieve("  ") == []
    assert retriever.retrieve("任意", limit=0) == []


def test_format_context_of_history(store):
    run_id = store.start_run("查天气")
    store.finish_run(run_id, final_answer="晴", status="success")

    retriever = MemoryRetriever(store=store, embedding_client=None)
    text = retriever.format_context(retriever.retrieve("查天气"))

    assert "查天气" in text
    assert "晴" in text


def test_format_context_empty():
    assert "没有检索到" in MemoryRetriever.format_context([])


# ---------- 历史对话：向量相关性召回 ----------

def _index(store, client, task, answer):
    """建一条成功 run 并写入向量，模拟「跑完一轮 → 建向量」。"""
    from memory.history import HistoryIndex

    run_id = store.start_run(task)
    store.finish_run(run_id, final_answer=answer, status="success")
    HistoryIndex(store=store, embedding_client=client).index_run(
        run_id, task, answer
    )
    return run_id


def test_retrieve_history_ranks_by_semantic_similarity(store):
    """改写/指代也能命中：靠向量相似度，而不是整句子串匹配。

    这里刻意用「计算」而不用「天气」：时效性问题现在被整体挡在历史注入之外，
    拿天气当样本就测不到相似度排序本身了。
    """
    client = FakeSemanticEmbeddingClient()
    calc = _index(store, client, "计算123*456", "123乘456等于56088")
    _index(store, client, "帮我写个快速排序", "好的，代码如下")

    retriever = MemoryRetriever(store=store, embedding_client=client)
    results = retriever.retrieve_history("123乘456是多少来着")

    assert [r["run_id"] for r in results] == [calc]
    assert results[0]["final_answer"] == "123乘456等于56088"
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_retrieve_history_skips_low_similarity(store):
    """不相关主题直接丢弃，不做「按时间取最新」的降级注入。"""
    client = FakeSemanticEmbeddingClient()
    _index(store, client, "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert retriever.retrieve_history("帮我写个快速排序") == []


def test_retrieve_history_respects_limit(store):
    client = FakeSemanticEmbeddingClient()
    for index in range(4):
        _index(store, client, f"计算123*456第{index}次", f"等于56088之{index}")

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert len(retriever.retrieve_history("123乘456", limit=1)) == 1
    assert len(retriever.retrieve_history("123乘456", limit=2)) == 2


def test_retrieve_history_dedupes_identical_tasks(store):
    """同一句话问过多次只留一条，避免两条预算被同一问题占满。"""
    client = FakeSemanticEmbeddingClient()
    for _ in range(3):
        _index(store, client, "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert len(retriever.retrieve_history("123乘456", limit=2)) == 1


def test_retrieve_history_skips_the_same_question_asked_again(store):
    """与当前问题一字不差的记录不召回：那是自我召回，不是历史知识。

    真实场景：用户把同一句话问第二遍，上一轮的 run 已经进了索引，
    它会以最高相似度占掉名额，还可能是上一轮编错的答案。
    """
    client = FakeSemanticEmbeddingClient()
    _index(store, client, "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert retriever.retrieve_history("计算123*456") == []
    # 换个问法就正常召回——被挡掉的只是「同一句话」
    assert len(retriever.retrieve_history("123乘456是多少")) == 1


def test_retrieve_history_skips_entries_without_usable_content(store):
    """只剩一句空摘要的条目不进候选：它既占名额又误导。"""
    client = FakeSemanticEmbeddingClient()
    run_id = _index(store, client, "计算123*456", "123乘456等于56088")
    # 模拟提炼出来的摘要为空：format_history 时没有可用内容
    store.save_turn_memory(run_id, summary="", facts=None)

    retriever = MemoryRetriever(store=store, embedding_client=client)
    results = retriever.retrieve_history("123乘456是多少")

    # 摘要为空时仍可用 final_answer 兜底，不该整条丢掉
    assert len(results) == 1


# ---------- 时效性问题：整体不注入历史 ----------


@pytest.mark.parametrize(
    "task",
    [
        "查询广州今日的天气，以可爱风格回答",
        "北京现在气温多少",
        "帮我看看今天的新闻",
        "茅台股价多少",
        "库里今天得了多少分",
        "What is the weather in Guangzhou?",
    ],
)
def test_time_sensitive_queries_skip_history(store, task):
    """天气/价格/比分这类问题的答案过期即错，宁可不注入。

    真实事故：用户问广州天气，注入块把上一轮成都的「14-19°C 阴天转小雨」
    排在最前，模型直接照抄成了广州的天气。
    """
    client = FakeSemanticEmbeddingClient()
    _index(store, client, "计算123*456", "123乘456等于56088")
    calls_before = len(client.calls)   # 建索引时已经打过一次

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert retriever.retrieve_history(task) == []
    # 连 embedding 都不该调用：省一次网络往返
    assert len(client.calls) == calls_before


@pytest.mark.parametrize(
    "task",
    ["123乘456是多少", "计算123*456的结果", "123×456 等于几"],
)
def test_stable_queries_still_get_history(store, task):
    client = FakeSemanticEmbeddingClient()
    _index(store, client, "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert len(retriever.retrieve_history(task)) == 1


def test_retrieve_history_excludes_failed_runs(store):
    client = FakeSemanticEmbeddingClient()
    failed = store.start_run("计算123*456")
    store.finish_run(failed, final_answer="", status="failed")
    store.save_run_embedding(
        failed, client.embed_query("计算123"), client.model, "v1"
    )

    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert retriever.retrieve_history("123乘456") == []


def test_retrieve_history_ignores_vectors_from_other_model(store):
    _index(
        store,
        FakeSemanticEmbeddingClient(model="old-model"),
        "计算123*456",
        "123乘456等于56088",
    )

    retriever = MemoryRetriever(
        store=store, embedding_client=FakeSemanticEmbeddingClient(model="new-model")
    )

    assert retriever.retrieve_history("123乘456") == []


def test_retrieve_history_without_client_returns_empty(store):
    """没有 embedding 客户端就不注入历史：不相关的历史是纯噪声。"""
    _index(store, FakeSemanticEmbeddingClient(), "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert retriever.retrieve_history("123乘456") == []


def test_retrieve_history_survives_embedding_failure(store):
    _index(store, FakeSemanticEmbeddingClient(), "计算123*456", "123乘456等于56088")

    retriever = MemoryRetriever(
        store=store, embedding_client=FakeSemanticEmbeddingClient(fail=True)
    )

    assert retriever.retrieve_history("123乘456") == []


def test_retrieve_history_guards_empty_query(store):
    client = FakeSemanticEmbeddingClient()
    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert retriever.retrieve_history("") == []
    assert retriever.retrieve_history("   ") == []
    assert retriever.retrieve_history("123乘456", limit=0) == []
    assert client.calls == []  # 空 query 不该打 embedding 服务


# ---------- 历史块格式化：条数与字符预算 ----------

def test_format_history_empty():
    assert MemoryRetriever.format_history([]) == ""


def test_format_history_includes_task_and_answer():
    text = MemoryRetriever.format_history(
        [{"task": "计算123*456", "final_answer": "56088"}]
    )

    assert "计算123*456" in text
    assert "56088" in text


def test_format_history_truncates_each_field():
    text = MemoryRetriever.format_history(
        [{"task": "问" * 500, "final_answer": "答" * 500}],
        task_chars=10,
        answer_chars=20,
    )

    assert len(text) < 120  # 单条被压到 10 + 20 + 模板文字
    assert "…" in text


def test_format_history_respects_max_chars():
    """字符预算是硬上限：超出立即停，不输出超长块。"""
    items = [
        {"task": f"任务{index}", "final_answer": "答" * 300}
        for index in range(5)
    ]

    text = MemoryRetriever.format_history(items, max_chars=200)

    assert len(text) <= 200
    assert "任务0" in text
    assert "任务4" not in text  # 预算不够，后面的条目根本不出现


# ---------- 偏好：目录（零 embedding）与按需加载 ----------



def test_preference_index_makes_no_embedding_call(store):
    """按需加载的前半段：目录不该为每轮对话付一次 embedding 请求。"""
    store.save_preference("u", "cat", "style", "回答简洁")
    client = FakeEmbeddingClient()
    retriever = MemoryRetriever(store=store, embedding_client=client)

    rows = retriever.retrieve_preference_index(query="随便问点什么", user_id="u")

    assert [row["memory_key"] for row in rows] == ["style"]
    assert client.calls == []


def test_preference_index_empty_without_any_preference(store):
    client = FakeEmbeddingClient()
    retriever = MemoryRetriever(store=store, embedding_client=client)

    assert retriever.retrieve_preference_index(query="q", user_id="u") == []
    assert client.calls == []


def test_preference_index_puts_newest_write_first(store):
    """最近写入的偏好必须排在最前：它是自动记忆唯一的注入入口。"""
    store.save_preference("u", "cat", "old", "a")
    store.save_preference("u", "cat", "new", "b")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.retrieve_preference_index(user_id="u")

    assert [row["memory_key"] for row in rows][0] == "new"


def test_preference_index_usage_breaks_tie_on_same_time(store):
    """写入时间完全相同时，被取用过的排前面。"""
    store.save_preference("u", "cat", "never_used", "a")
    used_id = store.save_preference("u", "cat", "often_used", "b")
    store.touch_preferences([used_id, used_id])
    with store._connect() as conn:
        conn.execute("UPDATE user_memories SET updated_at = '2026-01-01T00:00:00+00:00'")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.retrieve_preference_index(user_id="u")

    assert [row["memory_key"] for row in rows][0] == "often_used"


def test_newly_learned_preference_is_visible_in_a_crowded_catalog(store):
    """回归：偏好条数超过目录上限时，新学到的不能被老条目挤到列表外。

    旧实现以 use_count 为首键，覆盖/新增的条目 use_count 都是 0，
    于是「越新的记忆越看不见」——学习闭环断在最后一步。
    """
    for i in range(6):
        store.save_preference("u", "cat", f"old_{i}", f"老偏好{i}")
    retriever = MemoryRetriever(store=store, embedding_client=None)
    # 老偏好都被 load 过（use_count=1），新偏好是 0
    for i in range(6):
        retriever.load_preferences(query=f"old_{i}", user_id="u")

    store.save_preference("u", "cat", "brand_new", "刚学到的偏好")
    rows = retriever.retrieve_preference_index(user_id="u", limit=5)

    assert [row["memory_key"] for row in rows][0] == "brand_new"


def test_updated_preference_stays_visible_after_overwrite(store):
    """改过值的偏好同样要留在目录里：覆盖不该把它的可见性清零。"""
    for i in range(6):
        store.save_preference("u", "cat", f"k{i}", f"值{i}")
    retriever = MemoryRetriever(store=store, embedding_client=None)
    for i in range(6):
        retriever.load_preferences(query=f"k{i}", user_id="u")

    store.save_preference("u", "cat", "k0", "更新后的值")
    rows = retriever.retrieve_preference_index(user_id="u", limit=5)

    assert [row["memory_key"] for row in rows][0] == "k0"


def test_count_preferences_matches_catalog_scope(store):
    store.save_preference("u", "cat", "global_key", "v1")
    store.save_preference("u", "cat", "proj_key", "v2", scope="proj")
    store.save_preference("u", "cat", "other_key", "v3", scope="other")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert retriever.count_preferences("u") == 3
    # 与目录口径一致：给定 scope 时把 global 算进来
    assert retriever.count_preferences("u", scope="proj") == 2


def test_preference_index_clamps_limit(store):
    for i in range(PREF_INDEX_MAX_ITEMS + 5):
        store.save_preference("u", "cat", f"k{i}", "v")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.retrieve_preference_index(user_id="u", limit=100)

    assert len(rows) == PREF_INDEX_MAX_ITEMS


def test_preference_index_scope_includes_global(store):
    store.save_preference("u", "cat", "proj_key", "v1", scope="proj")
    store.save_preference("u", "cat", "global_key", "v2", scope="global")
    store.save_preference("u", "cat", "other_key", "v3", scope="other")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    keys = {
        row["memory_key"]
        for row in retriever.retrieve_preference_index(user_id="u", scope="proj")
    }

    assert keys == {"proj_key", "global_key"}


# ---------- 偏好目录：同维度归并 + 低置信度过滤 ----------

def test_preference_index_merges_same_dimension(store):
    """回归：同一件事存了三个 key，目录里必须只出现一条。

    真实数据里 ``detail_level：简短`` / ``detail_degree：short`` /
    ``response_style：简短`` 三条并存，目录列 5 行、模型能用的只有 1~2 条。
    """
    store.save_preference("u", "response_style", "detail_level", "简短", confidence=0.7)
    store.save_preference("u", "detail_degree", "detail_degree", "short", confidence=0.8)
    store.save_preference("u", "response_style", "response_style", "简短", confidence=0.8)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.retrieve_preference_index(user_id="u")

    assert [row["memory_key"] for row in rows] == ["answer_length"]
    # 同维度取值最高（并列取最近写入）的那条胜出
    assert rows[0]["memory_value"] == "简短"
    assert retriever.count_preferences("u") == 1


def test_preference_index_drops_low_confidence_noise(store):
    """置信度明显是「模型猜的」的条目不该进目录、也不该占条数。"""
    store.save_preference("u", "response_style", "language", "english", confidence=0.1)
    store.save_preference("u", "response_style", "answer_length", "简短", confidence=0.8)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.retrieve_preference_index(user_id="u")

    assert [row["memory_key"] for row in rows] == ["answer_length"]
    assert retriever.count_preferences("u") == 1


def test_preference_index_trims_redundant_category(store):
    """category 照抄 key 或等于该维度的默认归类时，不再印到每一行前面。"""
    store.save_preference("u", "detail_degree", "detail_degree", "short", confidence=0.9)
    store.save_preference("u", "response_style", "language", "中文", confidence=0.9)
    store.save_preference("u", "tooling", "python_cmd", "用 uv 跑", confidence=0.9)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    text = retriever.format_preference_index(
        retriever.retrieve_preference_index(user_id="u")
    )

    assert "- answer_length：简短" in text      # category 无信息量 → 省掉
    assert "- language：中文" in text
    assert "- [tooling] python_cmd：用 uv 跑" in text  # 真分类保留
    assert "[detail_degree]" not in text
    assert "[response_style]" not in text


def test_load_preferences_by_canonical_key_matches_alias(store):
    """目录里显示的是规范 key，按它点名加载时必须能命中库里的旧 key。"""
    store.save_preference("u", "response_style", "detail_degree", "short")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    for key in ("answer_length", "detail_degree", "detail_level"):
        rows = retriever.load_preferences(keys=[key], user_id="u")
        assert [row["memory_key"] for row in rows] == ["answer_length"], key


def test_load_preferences_by_key_ignores_confidence_threshold(store):
    """点名要的条目不受目录阈值拦截：模型说要看这条就得给。"""
    store.save_preference("u", "response_style", "language", "english", confidence=0.1)
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.load_preferences(keys=["language"], user_id="u")

    assert [row["memory_value"] for row in rows] == ["英文"]


def test_format_preference_index_clips_long_values():
    rows = [{"memory_key": "k", "category": "c", "memory_value": "x" * 200}]

    text = MemoryRetriever.format_preference_index(rows)

    assert text.startswith("- [c] k：")
    assert text.endswith("…")
    assert len(text) <= PREF_INDEX_VALUE_CHARS + len("- [c] k：") + 1


def test_format_preference_index_renders_structured_values():
    rows = [{"memory_key": "k", "category": "c", "memory_value": {"b": 2, "a": 1}}]

    text = MemoryRetriever.format_preference_index(rows)

    assert '{"a": 1, "b": 2}' in text


def test_format_preference_index_respects_max_chars():
    rows = [
        {"memory_key": f"k{i}", "category": "c", "memory_value": "v" * 100}
        for i in range(10)
    ]

    text = MemoryRetriever.format_preference_index(rows, max_chars=60)

    assert len(text) <= 60


def test_format_preference_index_empty():
    assert MemoryRetriever.format_preference_index([]) == ""


def test_load_preferences_by_key_skips_embedding(store):
    store.save_preference("u", "cat", "style", "回答简洁")
    store.save_preference("u", "cat", "lang", "中文")
    client = FakeEmbeddingClient()
    retriever = MemoryRetriever(store=store, embedding_client=client)

    rows = retriever.load_preferences(keys=["style"], user_id="u")

    assert [row["memory_key"] for row in rows] == ["style"]
    assert client.calls == []


def test_load_preferences_matches_key_substring(store):
    store.save_preference("u", "cat", "preferred_detail_level", "简洁")
    client = FakeEmbeddingClient()
    retriever = MemoryRetriever(store=store, embedding_client=client)

    rows = retriever.load_preferences(query="detail", user_id="u")

    assert [row["memory_key"] for row in rows] == ["preferred_detail_level"]
    assert client.calls == []


def test_load_preferences_falls_back_to_semantic_search(store):
    from memory.preference_manager import PreferenceManager

    client = FakeSemanticEmbeddingClient()
    manager = PreferenceManager(store=store, embedding_client=client)
    manager.upsert("u", "weather_style", "查天气时给穿衣建议")
    manager.upsert("u", "math_style", "算数时给步骤")
    retriever = MemoryRetriever(store=store, embedding_client=client)

    rows = retriever.load_preferences(query="今天气温多少", user_id="u")

    assert "weather_style" in [row["memory_key"] for row in rows]
    assert client.calls  # 语义兜底确实用了一次 embedding


def test_load_preferences_without_query_lists_all(store):
    store.save_preference("u", "cat", "k1", "v1")
    store.save_preference("u", "cat", "k2", "v2")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    rows = retriever.load_preferences(user_id="u")

    assert {row["memory_key"] for row in rows} == {"k1", "k2"}


def test_load_preferences_marks_usage(store):
    memory_id = store.save_preference("u", "cat", "style", "简洁")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    retriever.load_preferences(keys=["style"], user_id="u")

    row = [r for r in store.get_preferences("u") if r["memory_id"] == memory_id][0]
    assert row["use_count"] == 1


def test_load_preferences_guards_bad_limit(store):
    store.save_preference("u", "cat", "style", "简洁")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    assert retriever.load_preferences(user_id="u", limit=0) == []


def test_load_preferences_survives_usage_count_failure(store, monkeypatch):
    """计数失败（老库没迁移/写锁）不能把读取也带崩。"""
    store.save_preference("u", "cat", "style", "回答简洁")
    retriever = MemoryRetriever(store=store, embedding_client=None)

    def boom(_ids):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(store, "touch_preferences", boom)

    assert [row["memory_key"] for row in retriever.load_preferences(user_id="u")] == ["style"]


def test_format_history_prefers_distilled_summary():
    """有提炼摘要时用它代替原始答案：同样预算能表达更多。"""
    items = [
        {
            "task": "写个快排",
            "final_answer": "三千字的完整实现与讲解" + "…" * 100,
            "summary": "给出了 Python 递归版快排，复杂度 O(n log n)。",
        }
    ]

    text = MemoryRetriever.format_history(items)

    assert "给出了 Python 递归版快排" in text
    assert "三千字的完整实现" not in text


def test_format_history_falls_back_to_answer_without_summary():
    items = [{"task": "t", "final_answer": "原始答案", "summary": None}]

    assert "原始答案" in MemoryRetriever.format_history(items)
