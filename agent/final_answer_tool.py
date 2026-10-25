"""``final_answer`` 的宽容版：接受多余的关键字参数并丢掉。

smolagents 在 ``MultiStepAgent.__init__`` 里用
``self.tools.setdefault("final_answer", FinalAnswerTool())`` 塞进一个只接受
单个 ``answer`` 参数的内置工具。签名本身没错，但本地小模型会往里加参数：

    final_answer(结果, answer_format="分条")

真实现场（「查今日北京到上海高铁班次」）：llama3.2 把注入的偏好目录当成
``final_answer`` 的参数，写成 ``final_answer(split_result, answer_format='分条')``，
撞上 ``TypeError: final_answer() got an unexpected keyword argument
'answer_format'``。前面 ``web_search`` 已经把结果拿回来了，却卡在最后一行，
一步都没交出去；模型下一步开始瞎试别的写法。

拒绝的代价（这一步白跑 + 模型继续乱试）远大于接受（多余参数本来就无意义，
答案照旧只看第一个位置参数）。所以这里把 ``**kwargs`` 收下丢掉。

description 也改写了一句：明确「格式偏好由你自己写进答案文本，不是本工具的参数」。
光靠宽容只让这一步不崩，模型的注意力仍在错的地方。
"""

import ast
from typing import Any

from smolagents import FinalAnswerTool


class LenientFinalAnswerTool(FinalAnswerTool):
    """``final_answer``，忽略多余关键字参数。"""

    description = (
        "Provides a final answer to the given problem. "
        "It takes exactly one argument: the answer text. "
        "How that text should be formatted, how long it should be, or what "
        "tone it should use is decided by YOU while writing it — "
        "output-format preferences are NOT arguments of this tool."
    )

    # smolagents 默认校验「forward 的参数名 == inputs 的 key」。
    # 这里刻意多一个 **_ignored，所以关掉这项校验（inputs 本身仍然生效，
    # system prompt 里展示的签名还是 final_answer(answer)）。
    skip_forward_signature_validation = True

    def forward(self, answer: Any, **_ignored: Any) -> Any:
        return answer


# ---------------------------------------------------------------------------
# 执行前改写：把 final_answer(...) 上「answer 以外」的关键字参数删掉。
#
# 只换工具不够。`evaluate_python_code` 在真正执行前会把 final_answer 换成
# 一个本地闭包 def final_answer(answer): ...，签名写死成单个位置参数。
# 也就是说无论 Tool.forward 多宽容，final_answer(x, answer_format="分条")
# 都会在闭包这一层 TypeError。所以必须在代码进入 executor 之前改写。
# ---------------------------------------------------------------------------


class _DropExtraFinalAnswerKwargs(ast.NodeTransformer):
    """删掉 ``final_answer(...)`` 调用里 ``answer`` 以外的关键字参数。"""

    def __init__(self) -> None:
        self.rewritten = 0

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id == "final_answer"):
            return node
        # answer= 是合法写法（闭包的参数名就叫 answer），保留；
        # ** 展开（kw.arg is None）里可能装着 answer，也保留——
        # 删掉会让 final_answer() 变成无参调用，报的错更糟。
        # 其余具名参数（answer_format=、response_style=…）对答案没有意义，丢弃。
        kept = [kw for kw in node.keywords if kw.arg in (None, "answer")]
        if len(kept) == len(node.keywords):
            return node
        node.keywords = kept
        self.rewritten += 1
        return node


def rewrite_final_answer_calls(code: str) -> str:
    """去掉 ``final_answer`` 的多余关键字参数；无需改写时原样返回。

    语法有问题（模型输出的代码块常坏）就直接原样交回去，让 executor
    自己报它原本该报的错——这里不负责修语法。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    rewriter = _DropExtraFinalAnswerKwargs()
    tree = rewriter.visit(tree)
    if not rewriter.rewritten:
        return code

    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # pragma: no cover - unparse 对合法 AST 不会失败
        return code


class FinalAnswerRewritingExecutor:
    """包住 ``agent.python_executor``，执行前先做上面的改写。

    只包 ``__call__``，其余属性（``send_tools``/``send_variables``/``state``）
    原样转发给内部 executor。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __call__(self, code_action: str):
        return self._inner(rewrite_final_answer_calls(code_action))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


__all__ = [
    "FinalAnswerRewritingExecutor",
    "LenientFinalAnswerTool",
    "rewrite_final_answer_calls",
]
