from smolagents import OpenAIServerModel

# from config.settings import (
#     MODEL_ID,
#     API_BASE,
#     API_KEY,
#     MAX_TOKENS,
#     TEMPERATURE,
# )


# def create_model():
#     return OpenAIServerModel(
#         model_id=MODEL_ID,
#         api_base=API_BASE,
#         api_key=API_KEY,
#         max_tokens=MAX_TOKENS,
#         temperature=TEMPERATURE,
#     )

from config.settings import (
    ACTIVE_MODEL,
    MODEL_PROFILES,
    MAX_TOKENS,
    TEMPERATURE,
)


def create_model():
    if ACTIVE_MODEL not in MODEL_PROFILES:
        raise ValueError(
            f"未知模型配置：{ACTIVE_MODEL}，"
            f"可选值：{list(MODEL_PROFILES)}"
        )

    config = MODEL_PROFILES[ACTIVE_MODEL]

    if ACTIVE_MODEL == "deepseek" and not config["api_key"]:
        raise ValueError("请在 .env 中配置 DEEPSEEK_API_KEY")

    return OpenAIServerModel(
        model_id=config["model_id"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        client_kwargs={
            "timeout": 60.0,
            "max_retries": 0,
        },
    )
