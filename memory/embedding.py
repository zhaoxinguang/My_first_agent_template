
import os
from openai import OpenAI


class EmbeddingClient:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        provider=None,
        timeout=60.0,
    ):
        self.provider = (
            provider or os.getenv("EMBEDDING_PROVIDER", "ollama")
        ).lower()

        if self.provider == "ollama":
            self.base_url = (
                base_url
                or os.getenv(
                    "EMBEDDING_BASE_URL",
                    "http://localhost:11434/v1",
                )
            )
            self.model = (
                model
                or os.getenv(
                    "EMBEDDING_MODEL",
                    "nomic-embed-text",
                )
            )
            self.client = OpenAI(
                api_key="ollama",
                base_url=self.base_url,
                timeout=timeout,
                max_retries=2,
            )

        elif self.provider == "openai":
            self.base_url = (
                base_url or os.getenv("EMBEDDING_BASE_URL")
            )
            self.model = (
                model
                or os.getenv(
                    "EMBEDDING_MODEL",
                    "text-embedding-3-small",
                )
            )
            key = api_key or os.getenv("EMBEDDING_API_KEY")
            if not key:
                raise ValueError(
                    "请设置环境变量 EMBEDDING_API_KEY"
                )

            kwargs = {
                "api_key": key,
                "timeout": timeout,
                "max_retries": 2,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self.client = OpenAI(**kwargs)

        else:
            raise ValueError(
                f"不支持的 embedding provider: {self.provider}"
            )

    def embed_query(self, text):
        if not text or not text.strip():
            raise ValueError("待向量化文本不能为空")

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts):
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("待向量化文本不能包含空字符串")

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [
            item.embedding
            for item in sorted(
                response.data,
                key=lambda item: item.index,
            )
        ]
