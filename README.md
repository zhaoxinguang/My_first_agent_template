---
title:my agent
emoji: ⚡
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
tags:
- smolagents
- agent
- smolagent
- tool
- agent-course
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## 项目结构

```
app.py                 入口：组装记忆 + agent + UI
agent/factory.py       创建 CodeAgent（模型 + 工具 + prompt）
agent/model.py         模型网关（ollama / llama / deepseek 三档 profile）
agent/answer_checks.py 最终答案硬校验：时效性数据必须本轮查证过
agent/guardrails.py    护栏装配：答案校验 + 旧 Observation 折叠 + 每轮重置
config/settings.py     全局配置与 .env 读取
config/prompts.yaml    CodeAgent 的 prompt 模板
tools/                 工具集（唯一入口 tools/__init__.py）
memory/                长期记忆：存储、向量、偏好、历史索引、检索、上下文拼装
Gradio_UI.py           对话界面 + run 记录
documents/             只读文档目录（read_text_file 的数据源）
data/                  SQLite 数据库
```

### 记忆调用链

```
用户输入
  → memory/agent.py::MemoryAgent.run
  → memory/context.py::build_augmented_task（检索 + 拼装，唯一注入点）
       ├─ 历史对话块：memory/history.py 建过向量的 run，按余弦相似度召回
       │              默认最多 2 条、整块 ≤400 字符（memory/retriever.py 的常量）
       └─ 偏好目录块：user_memories 里有效偏好的「key + 要点」，纯列表无说明文字，
                      完整取值由 load_user_preferences 工具按需取
  → agent/factory.py::create_agent 产出的 CodeAgent 执行
      ├─ 需要完整偏好 → load_user_preferences(query)
      ├─ 判断这条值得长期记 → save_user_preference(key, value)
      └─ 偏好过时/写满预算 → forget_user_preference(key)
  → Gradio_UI 记录 run（agent_runs），并立即建一条历史向量（run_embeddings）
```

偏好没有自动提炼：写不写由模型在任务收尾时自己决定（见「偏好写入」一节）。

两类记忆都走同一套 embedding（`.env` 里的 ollama / `nomic-embed-text`），
所以「当前任务」和「历史问题」「偏好」在同一个向量空间里比较：

| 环节 | 位置 | 说明 |
|---|---|---|
| 写入历史向量 | `Gradio_UI.py::save_final_answer` | 答案先落库，再建向量；建向量失败只丢向量，不丢记录 |
| 补建历史向量 | `app.py::backfill_history_index` | 启动时只补「当前模型还没有向量」的 run，幂等 |
| 历史召回 | `MemoryRetriever.retrieve_history` | 只召回成功且有答案的 run；低于相似度阈值（`HISTORY_MIN_SIMILARITY = 0.6`）丢弃；同一问句去重；**时效性问题直接不召回**；跳过与本轮问句一字不差的自我召回；没有可用内容的空条目跳过；有本轮摘要时优先用摘要 |
| 条数/字数预算 | `memory/retriever.py` 顶部常量 | `HISTORY_MAX_ITEMS=2`、`HISTORY_MAX_CHARS=400`，拼装层 `min(limit, 上限)` 强制收敛 |

历史块自带「仅供参考、不要照搬旧答案」的约束语；没有 embedding 服务时
历史召回直接返回空（不相关的历史比没有历史更糟），偏好则退化成结构化读取。

### 时效性事实：宁可少给历史，也不给过期答案

发生过一次真实事故：用户问「查询广州今日的天气」，本地小模型第二步直接
`final_answer("广州今日天气：阴天转小雨，气温14-19°C")`——这串数字是**上一轮成都**的。
`web_search("广州 今天 天气")` 本身完全正常（返回真实数据：广州 34°C 晴），模型压根没查。

根因不是幻觉，是**历史注入把旧答案顶到了最前面**，而当时的注入提示语写着
「已包含答案就直接基于它作答，不要重复搜索」。三层一起改：

| 层 | 位置 | 做法 |
|---|---|---|
| 召回前 | `memory/retriever.py::is_time_sensitive` | 天气/气温/股价/汇率/比分/新闻/航班… 命中关键词就**完全不召回历史**（连 embedding 都不调）。误判代价只是少一条辅助；漏判代价是直接给出错误答案 |
| 召回时 | `MemoryRetriever.retrieve_history` | ① 跳过与本轮问句规范化后**一字不差**的自我召回（它只是「上一轮问过」，不含新信息）；② 跳过摘要和答案都为空的空条目；③ 每条历史尾部追加「（记录于 2026-10-21 06:13）」，让模型自己判断新旧 |
| 拼装时 | `memory/context.py::HISTORY_INSTRUCTION` | **禁则写在最前面**：「凡是会随时间变化的事实（天气、价格、比分、新闻、日程、库存等），一律不要沿用历史，必须重新调用工具查询」。旧版本把这条写在整段最后，本地小模型读不到 |
| 提交答案时 | `agent/answer_checks.py::FreshnessCheck` | 代码级硬校验（见下），提示词治不住的由它兜 |

`build_history_context` 里还留了一道 `is_time_sensitive` 拦截，防止有人换掉
retriever 后绕过召回层的早退。

#### 答案硬校验（`final_answer_checks`）

smolagents 1.13 支持 `final_answer_checks`：提交 `final_answer` 时调用
`check(final_answer, memory)`，抛 `AssertionError` → 包装成 `AgentError` →
`_run` 里 `except AgentError: action_step.error = e`。**任务不中断**，
错误文本作为 Observation 回喂，模型下一步可以自己纠正。

判定规则：

1. 任务命中「时效主题词」（`CURRENT_FACT_TOPICS`：天气/气温/股价/比分/新闻/价格…）——
   这里比召回层的 `TIME_SENSITIVE_HINTS` **更窄**，不收「今天 / 现在 / 最新」这类时间副词，
   否则「今天帮我写个装饰器」也会被要求联网；
2. 且答案给出了具体数值或天气结论（`asserts_concrete_value`：有数字，或含晴/阴/雨/雪…）——
   「我查不到实时天气」这种诚实兜底不拦；
3. 且本轮**没有**调用过 `web_search` / `visit_webpage` → 拦。

两个实现要点：

- **取证不看 `memory.steps`**。`_validate_final_answer` 在 `_finalize_step` **之前**执行，
  当前这一步还没进 memory；模型在同一段代码里 `r = web_search(...)` 后立刻
  `final_answer(r)` 会被误判成「没查过」。所以用 `ToolCallLedger`：在联网工具的实例上
  挂一层 `forward` 记录，调用发生时就记账（不改工具定义、不改返回值）。
  同一条规则也适用于「一步都没跑通」守卫——它原本只数 `memory.steps` 里成功的步数，
  一步到位时同样恒为 0（见下面的事故复盘）。
- **有放行上限**（`MAX_REJECTIONS=2`）。模型可能反复给同一个答案，无限拦截会把
  `max_steps` 耗光，最后走兜底答案反而更糟。拦够次数就放行。

账本和计数器在每个 `agent.run()` 前清空——Gradio 后台是同一个 agent 实例反复跑，
不重置的话上一轮的 `web_search` 会替这一轮背书。

**兜底路径要单独守**。`final_answer_checks` 只作用于模型主动提交的 final_answer；
`max_steps` 用尽时走的是 `_handle_max_steps_reached` → `provide_final_answer`，
**不经过校验**。实测 llama3.2 跑「广州今日天气」时正好走这条路：7 步全在解析偏好
JSON 时报错，一次 `web_search` 都没调，兜底答案却写着「阴天，温度在22℃左右」。
所以 `attach_answer_checks` 额外包了一层 `provide_final_answer`：兜底答案若同样违规，
就在前面加 `⚠️ 未核实：（「天气」相关数据本轮没有查证）`——已经不剩下一步可以自纠，
只能把风险明写给用户看，而不是让它冒充查证过的结果。

### 上下文预算与工具纪律：把「慢」和「反复查」压下去

本机的核心约束是**生成速度**（llama3.2 约 15 tok/s、qwen3.5:9b 约 5 tok/s，且 98% 在 CPU），
而 CodeAgent 每一步都要把**全部历史 Observation 重新 prefill 一遍**。
所以「一次调用往上下文里塞多少字」直接等于秒数。四件事一起改：

| 手段 | 位置 | 做法 |
|---|---|---|
| 搜索结果瘦身 | `tools/web_tools.py::DeepSeekSearchTool` | 条数 10→3、单条摘要 600~800→240 字符、整次输出硬上限 1800 字符（`WEB_SEARCH_*`） |
| 按需抽取 | `tools/web_tools.py::ExtractTool`（`extract(url, question)`） | 抓页面但**只回传与问题相关的行**（≤1200 字符），替代「整页 10000 字符」；visit_webpage 同时也降到 6000。候选集来自 `_readable_text()` 清洗后的 HTML 正文（丢掉 script/nav/footer 和 `<title>`、链接只留文字），不是 markdown；命中块两侧紧邻的「数据块」（`14 阴 湿度 100%`）会一起带上。关键词全落在导航上时（问天气，页面上只有 `今日天气提示`、`7天预报`），改用「以命中块为中心的一段**连续**数据」，多个候选窗口取最靠前的首屏那段；纯导航标签块（`7天预报`、`24小时预报`、`省份表`、`温度`）最后统一剔除 |
| 抽取怎么判「数据」 | `_UNIT_VALUE_RE` / `_STATE_WORD_RE` / `_is_nav_label()` | **不限于天气**：`数字+单位`（`53%`、`4级`、`12.5 元`、`120ms`、`¥ 68`、`3 年`）就算数据值，纯状态词（晴/小雨/涨/跌/胜/已结束）单独成块也算。判据要求**块短**（≤24 字）：新闻标题里也常带 `13℃`，那是正文不是数据。词表可用 `WEB_EXTRACT_STATE_WORDS` 追加领域状态词（如 `议价|无货|满仓`） |
| 抽取怎么对关键词 | `_normalize_for_match()` | 比对前先做**同义归一**（今天→今日、明天→明日、气温→温度、降雨→降水），只影响匹配、不改写返回给模型的原文。页面写「今天 晴 25℃」、问题问「今日天气」时也能命中——否则整张卡片得 0 分，退化成整页兜底 |
| 兜底策略 | `_looks_like_chrome()` / `_strip_nav_lines()` | 一个词都没命中才兜底：先找数据最密的一段，再退回正文开头并剔除开头的菜单项。**「短但是正确答案」不会被当成导航壳**（`用 Depends 声明参数即可` 只有十几个字、没有数字，但有句读标点，判为正文），否则短答案会被换成整页噪声 |
| 重复调用拦截 | `TaskScopedCache` | 同一 query / 同一 url 本轮第二次调用**不发网络请求**，直接回缓存精简版 + 一句「不要重复搜索」；`extract` 与 `visit_webpage` 共享页面缓存，换个问题问同一页只重做抽取 |
| 调用次数上限 | 同上 `TaskScopedCache.allow()` | 换个措辞就能绕过去重，所以再加硬刹车：每个任务 `web_search` ≤4、`extract` ≤3、`visit_webpage` ≤2 次（`WEB_*_CALLS_PER_TASK`）。到顶后工具只回「用已有结果作答」 |
| 旧步骤折叠 | `agent/guardrails.py::ObservationTrimmer` | 只保留最近 2 步的完整 Observation，更早的截到 400 字符（`AGENT_KEEP_RECENT_OBSERVATIONS`），原始记录仍完整保存在 `memory.steps` 里 |

折叠的实测效果（每步 2000 字符输出、7 步，脚本话模型，不计网络）：

```
无护栏 每次请求的 prompt 字符数: [8705, 11779, 14853, 17927, 21001, 24075, 27149]
有护栏 每次请求的 prompt 字符数: [8705, 11779, 14853, 15854, 16855, 17856, 18857]
```

越到后面差得越多：第 7 步省 8.3k 字符 ≈ 5.5k token ≈ 本机 1 分多钟的纯 prefill。

同一句「查询广州今日的天气，以可爱风格回答」在 llama3.2 上的实测对比
（`AGENT_MAX_STEPS=4`，真实 Ollama + 真实搜索接口）：

| | 改前 | 改后 |
|---|---|---|
| 总耗时 | 325 s | **131 s** |
| 步数 | 7（打满 max_steps 走兜底） | 4（正常收尾，没打满） |
| `web_search` 调用 | 7 次（每次换措辞，全发网络） | **2 次** |
| 答案 | 「晴天，温度 22℃ 左右」（22 是编的） | 「晴天，33-21℃」（数字都能在搜索结果里对上） |

要注意：**这些改动治的是「重复劳动」和「上下文膨胀」**，不是模型本身的代码能力——
llama3.2 仍会在 `response_style = load_user_preferences(...)` 这类地方写崩几步。

#### 答案依据校验（比「有没有查过」更细一层）

`FreshnessCheck` 在原来的「时效性数据必须本轮查证过」之外又加了两条：

1. **数字要有依据**：答案里 ≥2 位（或带小数）的数字，必须出现在本轮工具输出里。
   实测 llama3.2 有时确实搜了，但仍然报出「14-19°C」这种没出现过的数字——
   这条专门抓它。只对「问此刻事实」的任务生效，算出来的结果（`123×456=56088`）不会被误伤。
2. **一步都没跑通就不许下结论**：本轮所有步骤都报错（典型是连续 KeyError / NameError /
   代码解析失败，最后基于一个不存在的变量给结论）而答案又给出了具体数值 → 拦。
   回答「我没查到」不拦。

三条规则共用一个放行上限（`MAX_REJECTIONS=2`），避免把 `max_steps` 耗光。

对应地，`config/prompts.yaml` 也加了三条硬规矩（11~13）：同一 query 只搜一次、
摘要不够就用 `extract` 而不是整页、报出的数字必须来自本轮某个 Observation。

### 用户偏好：目录注入 + 按需加载 + 动态更新

偏好是最容易失控的一类记忆：条数只增不减，最后要么把上下文吃光，
要么每条都被截断成不可用的碎片。所以这里把它拆成「便宜的一半」和「按需的一半」：

| 环节 | 位置 | 说明 |
|---|---|---|
| 目录注入 | `MemoryRetriever.retrieve_preference_index` | **不调 embedding**：先数条数，为 0 直接不注入；有偏好也只给 `key + 要点`（≤48 字符），最多 8 条、整块 ≤620 字符 |
| 目录排序 | 同上 | 按「最近写入 → 被取用次数」排（`updated_at` 为首键）。query 不参与排序：目录每轮进 prompt，排序漂移对 prompt cache 和模型记忆都不友好 |
| 目录截断提示 | `memory/context.py::_build_preference_block` | 目录被上限截断时只补一行 `- （另有 N 条偏好，需要时用 load_user_preferences("") 查看）`——仍然保持列表形态，不写整段说明 |
| 目录形态 | 同上 | prompt 里就是纯列表 `- key：value`，没有任何说明文字；「怎么按需取全文」「什么时候该写偏好」写在 `config/prompts.yaml` 的固定规则 14~16 里，不在每轮注入块里重复 |
| 按需加载 | `MemoryRetriever.load_preferences` | 三种入口：点名 key（子串匹配，不走向量）→ 语义查找（走向量，服务挂了退化结构化）→ 列全部；命中后记一次 `use_count` |
| 动态更新 | `PreferenceManager.upsert` | 同 `(user, scope, key)` 再写即覆盖，旧值保留为 inactive 历史版本 |
| 预算约束 | `PreferenceManager._check_budget` | 单用户总字数 3500 / 40 条（`PREF_CHAR_LIMIT`、`PREF_MAX_ITEMS`）；超限**报错**而不静默截断，错误信息里带「当前占用 / 上限 / 该删哪条」 |
| 删除 | `PreferenceManager.forget` | 按 key 删除，global 找不到再看该用户全部作用域，避免静默失败 |
| 使用信号 | `MemoryStore.touch_preferences` | 只加 `use_count` / `last_used_at`，不动 `updated_at`：用过的偏好不该被当成「刚被改写」 |
| 覆盖继承 | `MemoryStore.save_preference` | 覆盖同 key 时新版本**继承**旧版本的 `use_count` / `last_used_at`：这两个字段描述「这条偏好」，不是「这一行版本」；不继承就会每改一次值清零一次，让刚更新的条目在目录里沉底 |
| 维度归并 | `memory/retriever.py::_catalog_rows` | 目录/计数/按需加载统一按**规范 key** 归并：`detail_level`、`detail_degree`、`verbosity` 只显示一条 `answer_length`；同一 key 多条取「置信度最高 → 最近写入」。库里堆着同义记录时，目录里不会再出现「5 条其实只有 2 个维度」 |
| 低置信度过滤 | 同上 (`PREF_INDEX_MIN_CONFIDENCE = 0.3`) | 置信度低于 0.3 的条目不进目录：那基本是模型「猜的」，还容易和真偏好并排打架 |
| 降级写入闸门 | `PreferenceManager.upsert` | 新值置信度比旧值低 0.2 以上时**拒绝覆盖**，抛 `PreferenceSkipped` 并把原因回给模型：`0.1` 的猜测不许顶掉用户明确说过的 `0.8`（真实事故，见下） |
| 归一化 | `memory/preference_manager.py::normalize_preference` | 纯函数，不碰数据库：① key 别名 → 五个规范维度（`detail_level`/`verbosity` → `answer_length`）；② 值写法归一（`short` → `简短`、`md` → `markdown`、`列表` → `分条`）；③ category 收敛到四个值。**只按 key 归一、不按值猜维度**——早期版本会把 `response_style：简短` 改写成 `answer_length：简短`，等于替模型猜语义。归一在 `upsert` 里调用，所以同义 key 写两次是覆盖、不是并列 |

**一次真实事故**（照抄进代码注释里当回归依据）：写侧曾在 `run 35` 写出
`language：english`（`confidence=0.1`），旧 `upsert` 不看置信度，直接把它写进库、
把 `language：中文`（`0.8`）顶成失效；之后每轮 prompt 都带着
`language：english`，写侧读到「已知偏好」继续输出 english，
置信度从 `0.1` 涨到 `0.9`——自增强闭环。现在剩下的两道闸：提示词固定 key 且
「只有用户明确要求用某种语言回答时才收 language」、`upsert` 的降级写入闸门，
以及目录侧的低置信度过滤（`PREF_INDEX_MIN_CONFIDENCE`）。

早期还有第三道「翻译型噪声过滤」（拦 `answer_format：informal` 这种把中文偏好
译成英文的脏值），已随反向纠错一起删掉：那是替小模型猜语义、且有误伤
（用户可能真的说 english）。代价是这类值现在会写进库，需要时用
`forget_user_preference` 清掉。

写侧入口有三个，最终都走同一条 `upsert` 路径：

1. agent 的 `save_user_preference` 工具（**唯一的生产入口**：模型在任务收尾时
   自行判断要不要记，见下一节）；
2. `forget_user_preference` 工具（删除，也用于腾预算）；
3. 手工排障：直接调 `PreferenceManager(store=...).upsert(...)` /
   `.forget(...)`（`set_preference.py` 脚本已删除，排障用这两种 API 更直接）。

所以「模型自己判断该记的偏好」和「手工写入的偏好」都在同一条预算下排队。

`last_used_at` / `use_count` 是后加的列，`MemoryStore._ensure_columns` 会在
启动时给老库幂等补列，不需要手工迁移。

### 偏好写入：由模型自己决断，没有自动提炼

**没有**每轮结束的自动提炼（早期的 `TurnDistiller` 已删除）。写不写、写什么，
由模型在任务收尾时自己判断并调用 `save_user_preference`：

| 环节 | 位置 | 说明 |
|---|---|---|
| 触发规则 | `config/prompts.yaml` 规则 16 | 调 `final_answer` 之前自问一次：本轮是否暴露了**下一轮仍然成立**的信息（语言 / 长度 / 格式 / 语气 / 工具环境 / 禁忌）？是则先 `save_user_preference` 再作答；否则一个字都不写 |
| 工具纪律 | 同上 | key 只能从 `language` / `answer_length` / `answer_format` / `response_style` / `tooling` 里选；value 用用户自己的语言写，几个词说清；不记任务进度、临时数据、一次性要求 |
| 写入归属 | `tools/memory_tools.py::SavePreferenceTool` | 工具描述里明写「系统不会自动提炼，是否写入由你自己判断」，避免模型以为有后台兜底而完全不写 |
| 归一与拦截 | `PreferenceManager.upsert` | key/value 归一、翻译噪声整条丢弃（返回 `None`）、低置信度不覆盖、预算校验，都在这里收口 |
| 工具回执 | `SavePreferenceTool.forward` | 被丢弃时返回 `Error: 未写入 …`，不能回「已保存」——否则模型以为写成功、下一轮找不到又写一遍 |
| 启动提示 | `app.py::log_preference_write_policy` | 启动日志里说明「无自动提炼，由模型自行判断」，避免「聊了很多轮却没有偏好」被当成 bug 排查 |

取舍：可控优先于自动。代价是偏好增长依赖模型自觉——本地小模型不会为了「下一轮」
付出当前这一步，所以实际很可能只在用户明说「记住」时才会写入。要恢复自动提炼，
需要重新引入一个每轮结束的后台提炼器（历史实现见 git 记录）。

`turn_memories`（本轮摘要 + facts）表仍然保留：存量摘要仍会被
`MemoryRetriever.format_history` 优先使用，只是不再自动产生新的。

## 工具清单

所有工具都通过 `tools/__init__.py` 注册，`build_tools()` 返回给 agent 的列表，
`describe_tools()` 打印「用途 + 调用方式 + 示例」。每个工具在 CodeAgent 眼里
就是一个可以直接写的 Python 函数：

| 工具 | 调用方式 | 用途 |
|---|---|---|
| `final_answer` | `final_answer(answer)` | 提交最终答案并结束任务，必须是最后一步 |
| `get_current_time_in_timezone` | `get_current_time_in_timezone("Asia/Shanghai")` | 查询指定时区当前时间 |
| `list_files` | `list_files()` | 列出 `documents/` 下可读的 .txt |
| `read_text_file` | `read_text_file("1.txt", start_line=1, end_line=20)` | 读取 `documents/` 下 .txt，支持行截取 |
| `load_user_preferences` | `load_user_preferences("response_style")` | 按需取偏好完整内容；传空字符串列全部 |
| `save_user_preference` | `save_user_preference("response_style", "回答简洁")` | 新增或覆盖长期偏好（带预算校验） |
| `forget_user_preference` | `forget_user_preference("response_style")` | 删除过时偏好，也是腾预算的手段 |
| `web_search` | `web_search("关键词")` | DeepSeek 官方搜索接口（`/v1/tools/search`），返回标题/链接/摘要。默认只回 3 条、总计 ≤1800 字符；同一 query 本轮重复调用走缓存。复用 `DEEPSEEK_API_KEY`，失败时返回 `Error:` 字符串 |
| `extract` | `extract(url, "广州今天气温多少度")` | 抓一个 URL 但**只回传与问题相关的句子**（≤1200 字符）。摘要不够时优先用它 |
| `visit_webpage` | `visit_webpage(url)` | 抓取网页全文转 Markdown（截断约 6000 字符）。只在确实要通读整页时用 |

偏好工具默认在 `build_tools()` 里注册，`app.py` 会把同一个 `store` 和
`embedding_client` 传给它——**工具与注入必须共用一个 store**，
模型刚 `save` 的偏好，下一轮目录才能立刻读到（这就是「动态更新」的落点）。

新增工具：在 `tools/` 下用 `@tool` 装饰函数或继承 `Tool`，然后在
`tools/__init__.py` 的 `build_tools()` 里追加，并在 `TOOL_GUIDE` 补一条说明。

## 启动

```bash
cp .env.example .env   # 填模型配置；本地可先起 ollama
pip install -r requirements.txt
python app.py          # 默认只监听本机，不生成公网链接
```

常用环境变量（都能写在 `.env` 里）：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `ACTIVE_MODEL` | `ollama` | 使用哪套模型配置（`ollama` / `llama` / `deepseek`）。三档都在 `config/settings.py` 的 `MODEL_PROFILES` 里 |
| `AGENT_MAX_STEPS` | `6` | 单次任务最大步数 |
| `AGENT_MAX_TOKENS` | `8192` | 单次**生成**的 token 上限（ollama 的 `num_predict`）。它突破不了 `num_ctx`，只写 Thought 不写代码块的报错见上面「排障」一节 |
| `AGENT_TEMPERATURE` | `0.5` | 采样温度 |
| `AGENT_REASONING_EFFORT` | `high`（开） | 思考开关：`none` 关闭，`low`~`max` 开启；留空=不传字段（ollama 上等于开）。见下面「思考模式」一节 |
| `AGENT_VERBOSITY_LEVEL` | `2` | 日志详细程度 |
| `AGENT_REQUEST_TIMEOUT` | `300` | 单次模型请求超时（秒）。本地 ollama 首次加载模型很慢，云端可调小到 60 |
| `AGENT_MAX_RETRIES` | `0` | 单次请求失败重试次数，云端网络抖动可设 `1`~`2` |
| `DEEPSEEK_SEARCH_BASE_URL` | 同 `DEEPSEEK_API_BASE` | `web_search` 用的 Search API 地址（走网关 / 私有部署时才需要填） |
| `PREF_CHAR_LIMIT` | `3500` | 单用户所有有效偏好的总字数上限，超了拒绝写入（`save_user_preference` 返回 `Error:`） |
| `PREF_MAX_ITEMS` | `40` | 偏好条数上限，防止预算被大量 20 字碎片切碎 |
| `GRADIO_SHARE` | `0` | 设为 `1` 才生成公网可访问链接 |
| `GRADIO_DEBUG` | `0` | 设为 `1` 开调试模式 |

> `share` 之前是硬编码 `True`，等于每次启动都把界面开到公网；现在默认关闭，
> 需要临时共享时才设 `GRADIO_SHARE=1`。

### 排障：小模型把偏好塞进工具参数

症状一，日志里出现：

```
Code execution failed at line 'x = load_user_preference(query='…')' due to:
InterpreterError: Forbidden function evaluation: 'load_user_preference' is not
among the explicitly allowed tools or defined/imported in the preceding code
```

工具真名是 `load_user_preferences`（复数），模型少写了一个 `s`。

症状二，同一类问题的另一种表现：

```
final_answer(split_result, answer_format='分条')
TypeError: final_answer() got an unexpected keyword argument 'answer_format'
```

模型把注入的偏好目录当成了一个**需要显式执行的子任务**——先去 `load_user_preference`
取偏好，再试图把 `answer_format` 传给 `final_answer`。前面 `web_search` 明明已经
拿到结果，却卡在最后一行，一步都没交出去。

两层处理：

1. **错误回执改写成可行建议**（`agent/guardrails.py` 的 `repair_tool_name_errors`）：
   把上面那串英文换成「没有名为 `load_user_preference` 的工具，最接近的是
   `load_user_preferences`。把这一行的函数名改成它后重新执行这一步；不要因为这行
   报错就跳过这个工具，也不要改用字符串切分之类的替代做法」。
   只改「发给模型的副本」，`memory.steps` 里保留原始报错可审计。
2. **`final_answer` 宽容化**（`agent/final_answer_tool.py`）：换掉内置工具，并包一层
   执行前的 AST 改写，删掉 `final_answer(...)` 上 `answer` 以外的关键字参数。
   只换工具不够——`evaluate_python_code` 执行前会把 `final_answer` 替换成只收一个
   位置参数的闭包，签名写死，所以必须在代码进 executor 之前改写。
   同时把「格式偏好由你自己写进答案文本，不是本工具的参数」写进工具 description。

剩余的根因（偏好块被当成待办事项）没有从提示词层面修：那需要往注入块里加说明文字，
与本项目「偏好块只保留 `- key：value` 列表」的口径冲突。若日志里仍频繁出现这类误用，
再考虑补一句。

### 排障：答案里全是链接和 `score`——模型把工具回执当答案交了

症状：`final_answer` 交出来的东西长这样（真实现场，查高铁班次）：

```
['本轮已经搜过这个 query，重复调用不会再请求网络。…',
 '…', 'DeepSeek 搜索返回 3 条结果（本次最多展示 3 条）：',
 '- G643次列车时刻表…', '  https://trains.ctrip.com/TrainSchedule/G643',
 '  携程火车票 · 2026-10-24 16:00:00 · score 2.1', '  您的位置：高铁动车 > …']
```

整份是 `web_search` 的原始回执被 `result.split('\n')` 拆开后原样交出，
`extract` 那一步的活儿一个字都没干。答案信息其实在摘要里
（`G643 | 北京南 | 上海虹桥 | 09:18 | 15:20`），但没有被整理出来。

**工具回执只放数据，不放行为指令。** 早期版本试过往回执里写「这不是答案」
「不要重复搜」「抓不到就换站点」，结果是：每次调用都重复一遍白白吃上下文，
而且小模型会把它们当成正文抄进答案——上面第一条 `'本轮已经搜过这个 query…'`
就是这么漏出去的。现在这些要求统一写在 `config/prompts.yaml` 的 web 规则
（规则 11~13：搜一次、摘要优先、元信息不许进答案），工具只说自己发生了什么：

```
搜索结果 2 条：
1. 海口天气预报
   链接：https://www.weather.com.cn/weather/101310101.shtml
   天气网 · 2026-10-24 16:00:00
   摘要：25日（今天） 多云 31/23℃ <3级 湿度 68%
2. Haikou - BBC Weather
   链接：https://www.bbc.com/weather/1809078
   摘要：Thundery showers
```

重复调用同一个 query 时只多一行事实：

```
（缓存：本轮已搜过同一个 query，未重新请求网络）
搜索结果 2 条：
…
```

具体改动：

- 标题行只报条数，不再附带「这不是答案」「先读摘要」之类的建议；
- `rerank_score` **不再渲染**。三条结果常常同分（日志里清一色 `2.1`），对模型没有
  区分度，却会作为 `score 2.1` 漏进答案；
- 条目从 `- ` 改成 `1.` 编号，模型可以引用「第 2 条」而不用复述 URL；
- 错误回执也只留 `Error:` 前缀 + 事实描述，不再跟「（这不是网页内容，不要拿它当
  正文解析）」「别在这个站上继续试」这类指令；
- 各种「已达上限」的刹车回执同样是纯事实，不再教模型下一步做什么。

> 顺带修掉一个截断 bug：重贴缓存结果时原来用 `_clip(cached, 800)` 按**字符**切，
> 而 cached 里的摘要行尾本来就带 `…（已截断 268 字）`，切点正落在标记中间，
> 输出成 `…（已…（已截断 397 字）`，两个数字还互相矛盾。改成 `_clip_lines()`
> 按整行切——标记总在行尾，不会被打断。

### 排障：模型放着摘要不用，一个个去抓天气站

症状：搜索摘要里明明已经有答案，模型却拿 URL 去抓页面，抓不到就换下一个站：

```
Step 1  web_search("Haikou current weather")   → 摘要里就有「25日（今天） 多云 31/23℃」
Step 2  extract(url=weather.com/...)            → Advertisement + 一行地名
Step 3  extract(url=theweathernetwork.com/...)  → 403
Step 5  web_search("Haikou weather")            → 又拿到一份带完整预报的摘要
Step 6  extract(url=bbc.com/weather/...)        → 超时
        Reached max steps.
```

两个原因叠在一起：

1. **回执文案把模型推去抓链接。** 上一版收尾写的是「要更精确的内容用 extract 抓上面的链接」，
   模型照做了。现在这句从工具回执里删掉，「摘要优先、抓不到就换站点」改成
   `config/prompts.yaml` 的规则 12——回执只给数据。
2. **抓到的只是一副页面骨架，工具却当成成功返回。** weather.com 这类站的正文由
   JavaScript 渲染，`requests` 只拿到 361 字全是标签（`Today` / `Hourly` /
   `Advertisement` / `Humidity`…），真实温度一个都没有。`extract_relevant_text`
   老实从中挑了 82 字交回去，模型看着像成功，于是继续换下一个天气站。

修法是新增骨架判据 `_looks_like_page_skeleton`，三个条件同时成立才拦：

| 条件 | 阈值 |
|---|---|
| 可读正文够短 | < 600 字 |
| 没有任何「数字+单位」或状态词 | `_UNIT_VALUE_RE` / `_STATE_WORD_RE` 都不命中 |
| 不像句子的片段占比过半 | ≥ 0.6 |

命中就返回 `Error: 页面只有导航骨架，正文由 JavaScript 渲染，静态请求拿不到。`
——只陈述发生了什么；「别在这个站上试、改用摘要作答」由 prompts.yaml 规则 12 负责。

两个坑要避开：**不能**用 `_UNIT_HINT_RE`（只要有数字就命中，`10-Day` 和邮编
`570312` 会把它放过去），也**不能**用 `_looks_like_chrome(fragment)` 数导航块
（它的单参语义是「整段结果像不像导航壳」，对所有短片段都返回 True，
一篇正常短文会被误判成骨架）。实测：

```
JS 骨架页   -> 骨架=True
天气内容页  -> 骨架=False
中文散文    -> 骨架=False
英文技术页  -> 骨架=False
```

### 排障：一步到位反而被判「一轮都没跑通」

症状：Step 1 的代码完全正确、`web_search` 和 `extract` 都成功回传，答案里的
数据也确实来自搜索结果，却被拦：

```
Check check_freshness failed with error: 本轮还没有任何一步成功执行
（每一步都报错或没有输出），你手上的变量可能根本不存在。…
```

根因：`failed_steps_problem` 用 `usable_steps(memory)` 判断「有没有跑通过」，
而 `final_answer` 是在**当前这一步写进 memory 之前**被校验的。模型一步到位
（同一段代码里 `web_search` → `extract` → `final_answer`）时，memory 里只有
system prompt 和 task 两条，这个计数恒为 0。

真实现场（「广州今日天气」）：Step 1 一次跑通，答案 29℃ / 湿度 52% 全部来自
搜索摘要，却被判「一步都没跑通」。模型信了这句话，后面 5 步重新搜索、拿回执
表头当关键词搜、把表头当 URL 传给 `extract`，越走越远——**一条判错的守卫
比一个跑错的模型更贵**。

修法：`failed_steps_problem` 在看 `memory` 之外也看账本，只要本轮有任一工具
回传过非 `Error` 的内容，就说明「手上的变量是存在的」，把数字依据交给
`grounding_problem` 逐项核对即可。全 `Error` 的场景仍然拦。

> 顺带：重复 query 的重贴预算从写死的 800 字改成跟随 `max_chars`（默认 1800）。
> 原来 800 会把刚拿到的第 3 条摘要整段削成「（重复的结果还有 2 行未显示）」，
> 模型反过来还得再搜一次。

### 排障：回答在半句话处断掉 / `code parsing` 失败

症状：日志里出现

```
Error in code parsing:
Your code snippet is invalid, because the regex pattern ```(?:py|python)?\n(.*?)\n``` was not found in it.
Here is your code snippet:
Thought: 用户想查询成都今天的天气，…
```

也就是**只有 Thought、没有代码块**，而且那句话往往以逗号结尾。

根因几乎总是 **Ollama 的上下文窗口（`num_ctx`）被 prompt 占满**，不是模型不会写代码，
`AGENT_MAX_TOKENS` 也管不到它：

| 参数 | 含义 | 默认 |
|---|---|---|
| `AGENT_MAX_TOKENS`（→ `num_predict`） | 最多**生成**多少 token | 8192 |
| `num_ctx` | prompt + 生成的**总窗口** | 4096（Ollama 默认） |

窗口剩下多少就只能写多少。实测一例：`Input-tokens: 3,986 + Output-tokens: 110 = 4,096`，
正好等于 `num_ctx`，135 秒只换来一句 45 字的话。

排查三步：

1. 看每步日志 `[model] step N | Input-tokens / Output-tokens`（stderr，也在 Gradio 脚注里）。
   `Input + Output` 顶到 2048/4096 就是窗口满了；
2. `ollama ps` 的 `CONTEXT` 列是实际生效的窗口大小；
3. 绕开 agent 直接打 Ollama 原生接口，看返回的 `done_reason`：
   `"length"` = 被长度截断，`"stop"` = 自然结束。

修法（**OpenAI 兼容端点会静默忽略请求里的 `num_ctx`**，所以只能在服务端改）：

```bash
ollama show <model> --modelfile | grep -i num_ctx   # 先确认 Modelfile 有没有写死
OLLAMA_CONTEXT_LENGTH=16384 ollama serve            # 没写死时这样生效
# Modelfile 里写死了就改 Modelfile 再 ollama create 一个新 tag
```

同时本项目已把「每轮注入块」压到约 1k token（system prompt 从 10.5k 字符收到 7.8k、
历史块 700→400 字符），单步 prompt 约 2.5~3k token，`num_ctx=8192` 即可留出足够生成空间。

### 思考模式（思维链）默认开启

Ollama 对支持思考的模型有一条很容易踩空的行为：**请求里不带任何思考字段时，它自动开思考**。
所以「我什么都没配，它为什么每轮都在思考」不是错觉。实测 qwen3.5:9b（问「17*23」，只要数字）：

| 请求 | completion_tokens | 正文 |
|---|---|---|
| 完全不传思考字段 | 293 | `391` |
| `reasoning_effort: "high"` | 294 | `391` |
| `reasoning_effort: "none"` | **4** | `391` |

但那 289 个思考 token 只在**裸问题**上出现。qwen3.5:9b 是否思考、思考多少，
完全取决于消息形状（本机实测，num_predict=600）：

| 消息形状 | 关思考 | 开思考（high） |
|---|---|---|
| 裸对话（连 system prompt 都没有） | 16 tok / 3.6s / 正文 25 字 | 600 tok / **144s** / **正文 0 字** |
| 真实首步（system prompt + 任务） | 76 tok / 18.2s / 正文 209 字 | 165 tok / 39.0s / 正文 174 字（思考 186 字） |
| 真实后续步（已有代码 + Observation） | 118 tok / 30.6s | 117 tok / 30.6s（思考 70 字） |

也就是说：

- **本项目不会走到裸对话那一行**——所有请求都带 `system_prompt`，里面的
  `Thought/Code` 示例本身就在压制思考；
- 开着思考的真实成本是**每轮首步多约 20 秒**（18s → 39s），后续步几乎无差别；
- 但这份"便宜"依赖消息形状。一旦哪天去掉 system prompt 里的格式示例，
  或换成不会读示例的模型，代价就是 144 秒换一个空回答。

想彻底消除这份不确定性就把 `AGENT_REASONING_EFFORT` 设成 `none`。
注意 `low` / `medium` 在 qwen3.5 上没有区别（它对 Ollama 是布尔型：
开就是开，三档都得到约 1900 字的思考），所以"少想一点"这个选项对它无效。

另外，smolagents 1.13 的 `OpenAIServerModel` 只取 `role/content/tool_calls`，
`reasoning_content` 直接被丢掉——**思考内容既不显示也不入库**，
所以它只影响耗时和窗口，不影响你能看到的东西。

这个开关对所有模型调用生效（agent 的每一步、以及你手工调 `create_model` 的地方）。
曾经有一处例外——每轮结束的自动记忆提炼固定用 `none`（它只要一段 JSON，
思考只会更容易把 JSON 截断）；自动提炼删除后，例外也一并消失了。

各端点的开关字段不一样，改之前先看这张表：

| 服务端 / 模型 | 关闭方式 | 坑 |
|---|---|---|
| Ollama + 思考模型（qwen3.5） | `reasoning_effort: "none"` | 原生的 `think: true/false` 在 `/v1` 端点上**被静默忽略**；`num_ctx` 同样被忽略 |
| Ollama + 非思考模型（llama3.2） | **不要发这个字段** | 实测会直接 **400**：`"llama3.2:latest" does not support thinking`，不是文档说的"无害忽略" |
| DeepSeek | 不传字段 | 它的枚举里没有 `none`，传了直接 400；不传就是不思考 |

`agent/model.py` 已经按档位处理了这个差异（`MODEL_PROFILES[...]["supports_thinking"]`），
并且加了兜底：万一档位声明支持、但实际模型不支持，第一次 400 后会自动去掉该字段
重试一次，并记住这个结论。你不需要自己记。

### 换模型 / 改配置为什么必须改 `.env`

`config/settings.py` 用 `load_dotenv(..., override=True)` 读项目根目录的 `.env`，
**`.env` 是唯一权威来源**。启动时会打印实际生效的档位和模型名：

```
[model] 生效档位：ACTIVE_MODEL=ollama → model_id=qwen3.5:9b @ http://localhost:11434/v1
```

为什么强调这一点：本目录的 `.envrc`（direnv）会把 `.env` 导出成 shell 环境变量。
按 dotenv 默认的 `override=False`（已有环境变量优先），你 `cd` 进目录那一刻
旧值就占了位，之后改 `.env` 完全无效——曾经的排查结论就是「改了 ACTIVE_MODEL
却没反应」。现在的口径：

- `ACTIVE_MODEL` 只选档位（`ollama` / `llama` / `deepseek`）；具体模型名在
  `OLLAMA_MODEL_ID` / `LLAMA_MODEL_ID` / `DEEPSEEK_MODEL_ID`；
  `llama` 档与 `ollama` 档共用同一个服务地址（`LLAMA_API_BASE` 默认跟随
  `OLLAMA_API_BASE`），只是模型名不同，方便在「质量」和「速度」之间切换：

  | 档位 | 默认模型 | 本机实测生成速度 | 冷加载 | 取舍 |
  |---|---|---|---|---|
  | `ollama` | `qwen3.5:9b` | 4.8 tok/s | 47s | 通用能力与代码质量最好 |
  | `llama` | `llama3.2:latest` | 9.7 tok/s | 15s | 快约 2 倍，写代码可靠性下降 |
  | `deepseek` | `deepseek-flash` | 云端 | — | 需 `DEEPSEEK_API_KEY` |

  （速度是本机纯 CPU 实测；`ollama ps` 显示 9B 模型只有约 2% 在显存，
  强制 `num_gpu=99` 反而降到 1.1 tok/s，所以这两个本地档都别开 GPU。）

- 改完 `.env` 只需重启 `python app.py`（模型在启动时构造一次，Gradio 不会重建）；
- 想让 shell 临时覆盖 `.env`，用 `env` 前缀不起作用，请改 `.env` 或临时改 `override`。

> `config/prompts.yaml` 必须完整提供 `system_prompt` / `planning` / `managed_agent` /
> `final_answer` 四段：smolagents 不会把传入的 `prompt_templates` 与内置默认模板合并，
> 缺段会在用到时才报 `KeyError`（`final_answer` 只在跑满 `AGENT_MAX_STEPS` 兜底作答时
> 才被读，最容易被漏掉）。`agent/factory.py` 会在启动时校验，缺段直接报错。


## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

用例全部离线可跑——不需要模型、不需要 ollama、不联网，也不碰
`data/agent_memory.db`（每个用例用独立的临时数据库）。覆盖范围：

- `test_store.py` 建表、run 生命周期、偏好版本化（同名 key 只留一条有效）、
  过期过滤、向量 upsert 与 scope 合并、老库补列迁移、使用次数计数与覆盖继承；
- `test_retriever.py` 余弦相似度的边界、向量排序、模型不匹配/服务不可用两条降级路径，
  以及偏好的「目录零 embedding」、「按需加载三种入口」、
  「目录被挤满时新学到/刚更新的偏好仍在最前」；
- `test_context.py` 偏好目录注入（长值只进要点）、「未命中就原样放行」、
  写完立刻生效（动态更新）、目录被上限截断时的那一行「另有 N 条」提示、
  以及「偏好块就是一列 `- key：value`、不含任何说明文字」；
- `test_preference_manager.py` 向量失败仍落库、`rebuild_embeddings` 统计、
  `upsert` 覆盖语义、字符/条数预算与按 key 遗忘、写侧归一（同义 key 只占一行）、
  翻译噪声整条丢弃（返回 `None`）；
- `test_memory_tools.py` 三个偏好工具：读、写、删、预算错误串、
  噪声值被丢弃时必须回 `Error:` 而不是「已保存」、懒建库（构造不碰 SQLite）；
- `test_memory_agent.py` 偏好注入位置、`stream/reset` 等参数透传、`create_agent` 组装；
- `test_file_tools.py` 只读 `documents/`、只允许 `.txt`、`../` 穿越与子目录拦截；
- `test_tools_registry.py` 工具清单与 `TOOL_GUIDE` 一致性、`final_answer` 必须在末尾；
- `test_embedding.py` / `test_settings.py` / `test_gradio_flags.py`
  默认配置、`health_check` 不抛异常、执行预算与 UI 开关的默认值。

## 配置与密钥

- 真实密钥只写在 `.env`，该文件已被 `.gitignore` 忽略；仓库里只有占位模板 `.env.example`。
- `.envrc`（direnv）不含任何明文，只负责把 `.env` 载入环境变量，本身也被忽略。
- `data/` 是运行期目录（SQLite 记忆库），不入库；首次运行 `MemoryStore` 会自动创建。
- 若密钥曾以任何形式提交或外泄，唯一有效的补救是到服务商控制台**吊销并重新签发**，
  再把新值写入本地 `.env`（改代码或删文件都无法让已泄露的旧密钥失效）。

## Embedding（偏好检索与历史召回）

程序**不会**替你启动 embedding 服务，也不会自动 pull 模型，它只连接下列外部服务：

| 环境变量 | 默认值 |
|---|---|
| `EMBEDDING_PROVIDER` | `ollama` |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` |
| `EMBEDDING_MODEL` | `nomic-embed-text` |

本地起 ollama 的做法：

```bash
ollama serve                      # 另开一个终端常驻
ollama pull nomic-embed-text      # 只需一次
python app.py                     # 启动时会打印 [memory] Embedding 服务可用：ollama / nomic-embed-text
```

启动日志就是唯一可靠的判断依据：`EmbeddingClient` 的构造函数不发网络请求，
所以「app 能起来」不代表服务可用。服务不可用时日志会明确打印
`Embedding 服务不可用（...）` 以及上述修复命令，此时：
- 偏好目录照常注入（目录不依赖向量），`load_user_preferences(语义查询)` 退化为结构化读取；
- 历史召回返回空（不相关的历史比没有历史更糟）。

换了 embedding 模型后，旧向量会因 `embedding_model` 不匹配被过滤，需要重建：

```bash
python -c "
from memory.store import MemoryStore
from memory.embedding import EmbeddingClient
from memory.preference_manager import PreferenceManager
print(PreferenceManager(MemoryStore(), EmbeddingClient()).rebuild_embeddings('default_user'))"
```

