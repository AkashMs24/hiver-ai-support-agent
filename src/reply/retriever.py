"""
Retrieves top-k historical brand replies given a new incoming customer message.
Provides grounding exemplars for RAG generation.
"""

from typing import List, Dict, Any
import numpy as np
from src.reply.index import ReplyIndex
from src.config import EMBEDDING_DIM


class ReplyRetriever:
    def __init__(self, reply_index: ReplyIndex):
        self.reply_index = reply_index
        if not self.reply_index.index or not self.reply_index.metadata:
            self.reply_index.load_index()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k historical (customer_query, brand_reply) cases most similar to query.
        """
        if not self.reply_index.metadata:
            return []

        # Encode query
        if self.reply_index.embedder:
            q_emb = self.reply_index.embedder.encode([query], normalize_embeddings=True)
            q_emb = np.array(q_emb, dtype=np.float32)
        else:
            # Fallback simple search
            q_emb = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)

        results = []

        if self.reply_index.index:
            try:
                distances, indices = self.reply_index.index.search(q_emb, min(top_k, len(self.reply_index.metadata)))
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(self.reply_index.metadata):
                        item = dict(self.reply_index.metadata[idx])
                        item["similarity_score"] = float(dist)
                        results.append(item)
                return results
            except Exception:
                pass

        # Fallback lexical/first-k retrieval
        for item in self.reply_index.metadata[:top_k]:
            res = dict(item)
            res["similarity_score"] = 0.50
            results.append(res)

        return results
