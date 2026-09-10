"""
Builds and maintains the FAISS vector index of brand historical replies.
Stores corresponding metadata (query, reply, intent, resolution) for grounded retrieval.
"""

import json
import os
from typing import List, Dict, Any
import numpy as np
from rich.console import Console

from src.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_DIM

console = Console()


class ReplyIndex:
    def __init__(self):
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedder = None
        self._init_embedder()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.embedder = None

    def build_from_conversations(self, conversations: List[Dict[str, Any]]):
        """
        Extract pairs of (customer_issue, brand_reply) and build FAISS vector index.
        """
        console.print(f"[bold blue]Indexing historical replies from {len(conversations)} conversations...[/bold blue]")

        items_to_index = []
        texts_to_embed = []

        for c in conversations:
            first_cust = c.get("first_customer_message")
            first_reply = c.get("first_brand_reply")

            if first_cust and first_reply and len(first_reply.strip()) > 10:
                texts_to_embed.append(first_cust)
                items_to_index.append({
                    "conversation_id": c.get("conversation_id"),
                    "customer_query": first_cust,
                    "brand_reply": first_reply,
                    "num_turns": c.get("num_turns", 1),
                })

        if not texts_to_embed:
            console.print("[yellow]Warning: No valid reply pairs found to index.[/yellow]")
            return

        # Compute embeddings
        if self.embedder:
            embeddings = self.embedder.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            # Fallback random/hash vectors for mock run if deps not yet installed
            embeddings = np.random.randn(len(texts_to_embed), EMBEDDING_DIM).astype(np.float32)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        # Build FAISS index
        try:
            import faiss
            index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product on normalized vectors = cosine similarity
            index.add(embeddings)
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            self.index = index
        except Exception as e:
            console.print(f"[yellow]FAISS library error ({e}), saving pure numpy matrix.[/yellow]")
            np.save(str(FAISS_INDEX_PATH).replace(".faiss", ".npy"), embeddings)

        self.metadata = items_to_index
        with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")

        console.print(f"[green]✓ Successfully built index with {len(items_to_index):,} historical brand responses[/green]")

    def load_index(self):
        """Load index and metadata from disk."""
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                import faiss
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            except Exception:
                pass

        if os.path.exists(FAISS_METADATA_PATH):
            self.metadata = []
            with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.metadata.append(json.loads(line))
