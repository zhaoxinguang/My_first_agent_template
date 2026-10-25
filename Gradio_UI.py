#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
import sys
from typing import Optional

from memory.store import MemoryStore
from memory.context import DEFAULT_USER_ID
from memory.history import HistoryIndex

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def _env_flag(name: str, default: bool = False) -> bool:
    """读取布尔型环境变量：1/true/yes/on 视为真，其余视为假。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = True,
    additional_args: Optional[dict] = None,
    on_finish=None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0
    step_seen = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            input_tokens = getattr(agent.model, "last_input_token_count", None) or 0
            output_tokens = getattr(agent.model, "last_output_token_count", None) or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = input_tokens
                step_log.output_token_count = output_tokens

            # 排障用：把每步的 token 数打到 stderr。
            # 本地 ollama 的 num_ctx（默认 4096）限制的是「prompt + 生成」的
            # 总窗口，不是 max_tokens。Input 一旦顶到窗口上限，模型会在半句
            # 话处停笔（末尾常是个逗号），表现为 code parsing 失败，
            # 而日志里没有任何异常。这一行让那种情况当场可查：
            # 例如 `Input-tokens:3,986 | Output-tokens:110` ≈ 4,096 = 窗口满了。
            print(
                f"[model] step {step_seen} | Input-tokens:{input_tokens:,} "
                f"| Output-tokens:{output_tokens:,}",
                file=sys.stderr,
                flush=True,
            )
            step_seen += 1

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    # final_answer = step_log  # Last log is the run's final_answer
    # final_answer = handle_agent_output_types(final_answer)
    final_answer = step_log.final_answer
    if on_finish is not None:
        on_finish(final_answer)
        
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(
        self,
        agent,  # 底层 CodeAgent，或带记忆注入的 MemoryAgent
        file_upload_folder: str | None = None,
        memory_store: MemoryStore | None = None,
        user_id: str = DEFAULT_USER_ID,
        history_index: HistoryIndex | None = None,
    ):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

        # UI 只负责记录 run，不做偏好检索与注入：
        # 那两件事统一由 memory/context.py + MemoryAgent 负责。
        self.memory_store = memory_store or MemoryStore()
        self.user_id = user_id
        # 历史对话的向量索引：给了就在每轮结束后补一条向量，
        # 这样后续对话的语义召回才有数据可查（没给就只存文本，不建向量）。
        self.history_index = history_index

    def interact_with_agent(self, prompt, messages):
        import gradio as gr

        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages

        # 1. 创建任务记录（记录用户原始输入，不是增强后的任务）
        run_id = self.memory_store.start_run(prompt)

        # 2. 保存任务最终结果
        def save_final_answer(answer):
            if isinstance(answer, (AgentText, AgentImage, AgentAudio)):
                answer = answer.to_string()
            else:
                answer = str(answer)

            self.memory_store.finish_run(
                run_id,
                final_answer=answer,
                status="success",
            )

            # 先落库文本，再建向量：embedding 服务抖动只会丢掉向量，
            # 不会丢掉「用户问了什么、模型答了什么」这条记录。
            if self.history_index is not None:
                self.history_index.index_run(
                    run_id=run_id,
                    task=prompt,
                    final_answer=answer,
                )

        # 3. 偏好注入由 MemoryAgent 统一完成，这里直接传原始 prompt
        try:
            for msg in stream_to_gradio(
                self.agent,
                task=prompt,
                reset_agent_memory=True,
                on_finish=save_final_answer,
            ):
                messages.append(msg)
                yield messages

        except Exception as exc:
            self.memory_store.finish_run(
                run_id,
                final_answer="",
                status="failed",
                error=str(exc),
            )
            raise

        yield messages



    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )

    def launch(self, share: bool | None = None, debug: bool | None = None, **kwargs):
        """启动界面。

        share/debug 默认从环境变量读取，缺省都是关闭：
        默认 share=True 会把界面直接暴露到公网，不该是默认行为。
        需要临时共享时用 ``GRADIO_SHARE=1 python app.py``。
        """
        import gradio as gr

        if share is None:
            share = _env_flag("GRADIO_SHARE", False)
        if debug is None:
            debug = _env_flag("GRADIO_DEBUG", False)

        with gr.Blocks(fill_height=True) as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                ),
                resizeable=True,
                scale=1,
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

        demo.launch(debug=debug, share=share, **kwargs)


__all__ = ["stream_to_gradio", "GradioUI"]