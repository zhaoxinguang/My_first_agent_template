"""测试专用的替身对象（不属于产品代码）。"""


class FakeEmbeddingClient:
    """可编程的 embedding 客户端，对外接口与 EmbeddingClient 一致。

    - model：用于验证「换模型后旧向量被过滤」；
    - fail：模拟 ollama 没起（embed_query 抛错）；
    - 向量由字符哈希生成，保证同一文本得到同一向量。
    """

    def __init__(self, model="fake-embed-v1", fail=False, dim=8):
        self.model = model
        self.fail = fail
        self.dim = dim
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        if self.fail:
            raise ConnectionError("stub: embedding 服务不可用")
        vector = [0.0] * self.dim
        for char in str(text):
            vector[hash(char) % self.dim] += 1.0
        return vector
