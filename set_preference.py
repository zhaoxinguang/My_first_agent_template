from memory.store import MemoryStore
from memory.embedding import EmbeddingClient
from memory.preference_manager import PreferenceManager

store = MemoryStore()
embedding_client = EmbeddingClient()

manager = PreferenceManager(
    store=store,
    embedding_client=embedding_client,
)

manager.save_preference(
    user_id="default_user",
    category="response_style",
    memory_key="preferred_detail_level",
    memory_value="回答简洁，优先提供分步骤操作和直接可用的代码",
    scope="global",
    confidence=0.95,
)
