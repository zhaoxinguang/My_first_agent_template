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


class FakeSemanticEmbeddingClient:
    """按「主题词」映射到固定向量的 embedding 替身。

    FakeEmbeddingClient 用字符哈希，只能保证「同文本 → 同向量」，
    没法稳定构造「A 比 B 更相关」。这个替身按主题出向量，
    相似度只由主题决定，方便断言相关性排序、阈值裁剪与去重行为。

    - 含「天气 / 气温」→ [1, 0, 0]
    - 含「计算 / 乘 / 123」→ [0, 1, 0]
    - 其它 → [0, 0, 1]
    """

    def __init__(self, model="fake-semantic-v1", fail=False):
        self.model = model
        self.fail = fail
        self.calls = []

    def _vector(self, text):
        text = str(text)
        weather = 1.0 if ("天气" in text or "气温" in text or "晴" in text) else 0.0
        math = 1.0 if ("计算" in text or "乘" in text or "123" in text) else 0.0
        other = 1.0 if (weather == 0 and math == 0) else 0.0
        return [weather, math, other]

    def embed_query(self, text):
        self.calls.append(text)
        if self.fail:
            raise ConnectionError("stub: embedding 服务不可用")
        return self._vector(text)
