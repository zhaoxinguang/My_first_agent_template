# from smolagents import CodeAgent,DuckDuckGoSearchTool,OpenAIServerModel,HfApiModel,load_tool,tool
# import datetime
# import requests
# import pytz
# import yaml
# from tools.final_answer import FinalAnswerTool
# import os
# from Gradio_UI import GradioUI
# from dotenv import load_dotenv
# from my_tools import read_text_file, list_files

# load_dotenv()


# @tool
# def get_current_time_in_timezone(timezone: str) -> str:
#     """A tool that fetches the current local time in a specified timezone.
#     Args:
#         timezone: A string representing a valid timezone (e.g., 'America/New_York').
#     """
#     try:
#         # Create timezone object
#         tz = pytz.timezone(timezone)
#         # Get current time in that timezone
#         local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
#         return f"The current local time in {timezone} is: {local_time}"
#     except Exception as e:
#         return f"Error fetching time for timezone '{timezone}': {str(e)}"

# search_tool = DuckDuckGoSearchTool()
# final_answer = FinalAnswerTool()

# # If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# # model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

# # model = HfApiModel(
# # max_tokens=2096,
# # temperature=0.5,
# # model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
# # custom_role_conversions=None,
# # )

# # model = OpenAIServerModel(
# #     model_id="deepseek-chat",
# #     api_base="https://api.deepseek.com",
# #     api_key=os.environ["DEEPSEEK_API_KEY"],
# #     max_tokens=2096,
# #     temperature=0.5,
# # )
# model = OpenAIServerModel(
#     model_id="qwen3.5:9b",
#     api_base="http://localhost:11434/v1",
#     api_key="ollama",
#     max_tokens=2048,
#     temperature=0.5,
# )

# # Import tool from Hub
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# with open("prompts.yaml", 'r') as stream:
#     prompt_templates = yaml.safe_load(stream)
    
# agent = CodeAgent(
#     model=model,
#     tools=[
#         get_current_time_in_timezone,
#         final_answer,
#         search_tool,
#         read_text_file,
#         list_files
#         ], ## add your tools here (don't remove final answer)
#     max_steps=6,
#     verbosity_level=2,
#     grammar=None,
#     planning_interval=None,
#     name=None,
#     description=None,
#     prompt_templates=prompt_templates
# )


# GradioUI(agent).launch()

from agent.factory import create_agent
from Gradio_UI import GradioUI
from memory.store import MemoryStore
from memory.retriever import MemoryRetriever
from memory.embedding import EmbeddingClient
from memory.preference_manager import PreferenceManager


def main():
    # 1. 初始化记忆存储
    memory_store = MemoryStore()

    # 2. 初始化远程 Embedding 客户端
    try:
        embedding_client = EmbeddingClient()
    except Exception as e:
        print(f"Embedding 初始化失败，将使用普通记忆读取：{e}")
        embedding_client = None

    # 3. 初始化记忆管理模块
    memory_retriever = MemoryRetriever(
        store=memory_store,
        embedding_client=embedding_client,
    )

    preference_manager = PreferenceManager(
        store=memory_store,
        embedding_client=embedding_client,
    )

    # 4. 创建原始 Agent
    agent = create_agent()

    # 5. 包装 Agent，让每次任务执行前都检索用户偏好
    memory_agent = MemoryAgent(
        agent=agent,
        retriever=memory_retriever,
        user_id="default_user",
    )

    # 6. 启动 Gradio
    GradioUI(memory_agent).launch()


class MemoryAgent:
    """在 Agent 执行前检索记忆，并注入任务上下文。"""

    def __init__(self, agent, retriever, user_id):
        self.agent = agent
        self.retriever = retriever
        self.user_id = user_id

    def run(self, task, *args, **kwargs):
        memories = self.retriever.retrieve_preferences(
            query=task,
            user_id=self.user_id,
            scope="global",
            limit=5,
        )

        memory_context = self.retriever.format_preferences(memories)

        augmented_task = (
            f"{memory_context}\n\n"
            "请在不违背当前任务要求的前提下，合理参考上述偏好。"
            "\n\n当前任务：\n"
            f"{task}"
        )

        return self.agent.run(
            augmented_task,
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    main()

