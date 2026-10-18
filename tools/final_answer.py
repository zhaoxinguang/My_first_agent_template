"""最终答案工具：agent 用它结束任务并返回结果。"""

from typing import Any

from smolagents.tools import Tool


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {
            "type": "any",
            "description": "The final answer to the problem",
        }
    }
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer


__all__ = ["FinalAnswerTool"]
