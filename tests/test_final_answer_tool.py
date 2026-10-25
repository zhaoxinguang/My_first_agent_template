"""``final_answer`` 的宽容版 + 执行前改写。

两个失败都来自同一个误读：模型把注入的偏好当成工具的**参数**。

真实现场（查今日北京到上海高铁班次，Step 1/2）：

    final_answer(split_result, answer_format='分条')
    → TypeError: final_answer() got an unexpected keyword argument 'answer_format'

前面 web_search 已经拿到结果，却卡在最后一行一步都没交出去。这里断言三件事：
宽容版工具本身不拒绝多余参数；进 executor 前参数就被删掉；少参数时照旧报错。
"""

import pytest

pytest.importorskip("smolagents")

from agent.final_answer_tool import (  # noqa: E402
    FinalAnswerRewritingExecutor,
    LenientFinalAnswerTool,
    rewrite_final_answer_calls,
)


# --- 工具本身 ---------------------------------------------------------------


def test_lenient_final_answer_ignores_extra_kwargs():
    tool = LenientFinalAnswerTool()

    assert tool("高铁：G1 07:00", answer_format="分条") == "高铁：G1 07:00"
    assert tool("答案", response_style="简短", answer_length="简短") == "答案"


def test_lenient_final_answer_keeps_the_declared_signature():
    tool = LenientFinalAnswerTool()

    # inputs 仍然只声明 answer：system prompt 里展示的签名不能变宽，
    # 否则等于告诉模型「可以传格式参数」。
    assert tool.name == "final_answer"
    assert list(tool.inputs) == ["answer"]


def test_lenient_final_answer_description_says_preferences_are_not_arguments():
    tool = LenientFinalAnswerTool()

    assert "NOT arguments of this tool" in tool.description


# --- 执行前改写 -------------------------------------------------------------


def test_rewrite_drops_extra_kwargs():
    assert (
        rewrite_final_answer_calls('final_answer(result, answer_format="分条")')
        == "final_answer(result)"
    )


def test_rewrite_keeps_answer_keyword():
    # 闭包的参数名就叫 answer，所以 answer= 是合法写法，不能改。
    code = 'final_answer(answer="结果")'
    assert rewrite_final_answer_calls(code) == code


def test_rewrite_leaves_clean_code_untouched():
    # 没被改写的代码要原样返回：unparse 会把引号、注释、空行全部重排，
    # 没必要让模型看到一份不像自己写的代码。
    code = 'x = 1\n# 注释\nfinal_answer("答案")'
    assert rewrite_final_answer_calls(code) == code


def test_rewrite_leaves_broken_code_untouched():
    # 模型输出语法坏掉时，让 executor 报它原本该报的错，这里不修语法。
    code = "final_answer("
    assert rewrite_final_answer_calls(code) == code


def test_rewrite_keeps_double_star_expansion():
    # ** 里可能装着 answer，删掉会变成无参调用，报的错比原来更难懂。
    code = 'final_answer(**{"answer": r})'
    assert rewrite_final_answer_calls(code) == code


def test_rewrite_ignores_unrelated_calls():
    code = 'result = web_search("高铁")\nfinal_answer(result)'
    assert rewrite_final_answer_calls(code) == code


# --- 端到端：过一遍真 executor ----------------------------------------------


class _FakeModel:
    model_id = "fake"


@pytest.fixture
def executor():
    """真 CodeAgent 的 python_executor，套上改写层。"""
    from smolagents import CodeAgent

    from tools.memory_tools import LoadPreferencesTool

    agent = CodeAgent(
        tools=[LoadPreferencesTool(store=None)],
        model=_FakeModel(),
        max_steps=1,
        verbosity_level=0,
    )
    agent.tools["final_answer"] = LenientFinalAnswerTool()
    wrapped = FinalAnswerRewritingExecutor(agent.python_executor)
    wrapped.send_tools({**agent.tools})
    return wrapped


def test_executor_no_longer_chokes_on_extra_kwargs(executor):
    """回归：这一行原本抛 TypeError，整步白跑。"""
    output, _logs, is_final = executor('final_answer("高铁 G1", answer_format="分条")')

    assert output == "高铁 G1"
    assert is_final is True


def test_executor_still_accepts_plain_call(executor):
    output, _logs, is_final = executor('final_answer("答案")')

    assert output == "答案"
    assert is_final is True


def test_executor_forwards_other_attributes():
    class _Inner:
        def __init__(self):
            self.state = {"_print_outputs": "logs"}
            self.sent = None

        def send_tools(self, tools):
            self.sent = tools

        def send_variables(self, variables):
            self.state.update(variables)

    inner = _Inner()
    wrapped = FinalAnswerRewritingExecutor(inner)

    wrapped.send_tools({"a": 1})
    wrapped.send_variables({"b": 2})

    assert inner.sent == {"a": 1}
    assert wrapped.state == {"_print_outputs": "logs", "b": 2}
