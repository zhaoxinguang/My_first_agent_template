"""复现：同一句天气任务连跑 N 次，看 llama3.2 每轮走的路径与最终答案。

只做观测，不改库：create_agent(store=None) 不写 agent_runs，
偏好工具只有被模型主动调用时才落库（观测里会看到）。
"""
import sys
import time

sys.path.insert(0, "/workspace")

from agent.factory import create_agent

TASK = "查询广州今日的天气"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def summarize(agent, run_index, elapsed, answer, error):
    steps = getattr(agent.memory, "steps", [])
    code_steps, errors = [], []
    for step in steps:
        code = getattr(step, "code_action", None)
        if code:
            code_steps.append(" ".join(str(code).split())[:160])
        err = getattr(step, "error", None)
        if err:
            errors.append(str(err).splitlines()[0][:160])
    print(f"\n=========== run {run_index}  ({elapsed:.1f}s) ===========")
    print(f"最终答案：{str(answer)[:300]!r}")
    if error:
        print(f"异常：{error}")
    print(f"步数：{len(code_steps)}  报错步：{len(errors)}")
    for i, code in enumerate(code_steps, 1):
        print(f"  step{i} code: {code}")
    for i, err in enumerate(errors, 1):
        print(f"  step{i} error: {err}")


for index in range(1, RUNS + 1):
    agent = create_agent(include_web=True)
    started = time.time()
    answer, error = None, None
    try:
        answer = agent.run(TASK)
    except Exception as exc:  # 记录而非中断，方便对比
        error = f"{type(exc).__name__}: {exc}"
    summarize(agent, index, time.time() - started, answer, error)
