"""
Data-driven intent discovery using Sentence-Transformers and HDBSCAN / K-Means clustering.
Helps uncover natural semantic clusters directly from historical raw inbound queries.
"""

import json
from typing import List, Dict, Any
import numpy as np
from rich.console import Console

from src.config import INTENT_CLUSTERS_PATH, EMBEDDINGS_CACHE_PATH

console = Console()


def discover_intents_from_data(
    messages: List[str],
    min_cluster_size: int = 25,
    sample_limit: int = 2500,
) -> Dict[str, Any]:
    """
    Cluster customer support messages to discover semantic topics.
    Uses sentence-transformers to embed texts and clustering algorithms to group them.
    """
    console.print(f"[bold blue]Running unsupervised intent discovery on {min(len(messages), sample_limit)} messages...[/bold blue]")

    subsample = messages[:sample_limit]

    # Try importing sentence_transformers
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        console.print("[cyan]Encoding messages into 384-dim semantic embeddings...[/cyan]")
        embeddings = embedder.encode(subsample, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:
        console.print(f"[yellow]SentenceTransformer fallback due to: {e}. Using TF-IDF representation...[/yellow]")
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=384, stop_words="english")
        embeddings = tfidf.fit_transform(subsample).toarray()

    # Save cached embeddings
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)

    # Perform clustering (Try HDBSCAN, fallback to KMeans)
    cluster_labels = []
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        cluster_labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        console.print(f"[green]✓ HDBSCAN discovered {n_clusters} clusters (noise points: {list(cluster_labels).count(-1)})[/green]")
    except Exception:
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = 12
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(embeddings)
        console.print(f"[green]✓ MiniBatchKMeans grouped messages into {n_clusters} clusters[/green]")

    # Extract top keywords and exemplar messages per cluster
    from sklearn.feature_extraction.text import TfidfVectorizer
    clusters_info = {}

    for c_id in set(cluster_labels):
        indices = [i for i, lbl in enumerate(cluster_labels) if lbl == c_id]
        if not indices:
            continue

        cluster_texts = [subsample[i] for i in indices]
        
        # Keyword extraction
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=8)
            vec.fit(cluster_texts)
            top_terms = list(vec.vocabulary_.keys())
        except Exception:
            top_terms = ["support", "device", "issue"]

        clusters_info[str(c_id)] = {
            "cluster_id": int(c_id),
            "size": len(indices),
            "top_terms": top_terms[:6],
            "exemplars": cluster_texts[:3],
        }

    discovery_result = {
        "num_analyzed": len(subsample),
        "num_clusters": len(clusters_info),
        "clusters": clusters_info,
    }

    with open(INTENT_CLUSTERS_PATH, "w", encoding="utf-8") as f:
        json.dump(discovery_result, f, indent=2)

    console.print(f"[green]✓ Intent discovery findings saved to {INTENT_CLUSTERS_PATH}[/green]")
    return discovery_result
