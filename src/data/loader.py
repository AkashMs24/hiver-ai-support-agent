"""
Dataset loader for the Customer Support on Twitter dataset.

Handles:
- Loading the raw CSV
- Filtering by brand
- Basic validation and statistics
"""

import json
import logging
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import RAW_CSV_PATH, BRAND, BRAND_STATS_PATH

logger = logging.getLogger(__name__)
console = Console()


def load_raw_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the raw Twitter Customer Support dataset.

    The dataset has columns:
    - tweet_id: Unique anonymized tweet ID
    - author_id: Unique anonymized user ID
    - inbound: Whether tweet is inbound (customer→brand)
    - created_at: Timestamp
    - text: Tweet content
    - response_tweet_id: IDs of response tweets (comma-separated)
    - in_response_to_tweet_id: ID of parent tweet

    Returns:
        DataFrame with all tweets
    """
    path = csv_path or str(RAW_CSV_PATH)
    console.print(f"[bold blue]Loading dataset from {path}...[/bold blue]")

    df = pd.read_csv(path)

    # Ensure correct types
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["author_id"] = df["author_id"].astype(str)
    df["inbound"] = df["inbound"].astype(bool)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)
    df["response_tweet_id"] = df["response_tweet_id"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    console.print(f"[green]✓ Loaded {len(df):,} tweets[/green]")
    return df


def identify_brand_author_ids(df: pd.DataFrame, brand_name: str) -> set:
    """
    Identify author IDs belonging to a brand.

    Brand agents are identified by:
    1. Their tweets are outbound (inbound=False)
    2. Their username appears in the text of customer tweets as @mention

    Since the dataset anonymizes @mentions to author_ids, we look for
    outbound authors who appear frequently.
    """
    # Outbound tweets are from brands
    outbound = df[~df["inbound"]]

    # Find the most prolific outbound authors — these are brands
    author_counts = outbound["author_id"].value_counts()

    # Look for the brand by checking if the brand name appears in responses
    # In the dataset, brand tweets often start with the customer's @mention
    brand_authors = set()

    # Strategy: find outbound authors whose tweets are responses to inbound tweets
    # mentioning the brand name in the author_id pattern
    for author_id, count in author_counts.head(100).items():
        # Get sample tweets from this author
        sample_tweets = outbound[outbound["author_id"] == author_id]["text"].head(5)
        # Check if any tweet text contains the brand name
        for text in sample_tweets:
            if brand_name.lower() in text.lower():
                brand_authors.add(author_id)
                break

    # Fallback: if brand detection via text fails, use the most common outbound authors
    # and look for author_id patterns
    if not brand_authors:
        # The dataset usually has brand_name as part of the author_id for major brands
        for author_id in author_counts.head(200).index:
            if brand_name.lower() in str(author_id).lower():
                brand_authors.add(author_id)

    # Final fallback: use top outbound authors as brands
    if not brand_authors:
        logger.warning(
            f"Could not identify {brand_name} by name. Using heuristic approach."
        )
        # Check which outbound authors respond to inbound tweets most
        # This is the most reliable heuristic
        inbound_tweet_ids = set(df[df["inbound"]]["tweet_id"])
        for author_id in author_counts.head(50).index:
            author_tweets = outbound[outbound["author_id"] == author_id]
            response_to = author_tweets["in_response_to_tweet_id"].dropna()
            # If most of their tweets are responses to inbound tweets, they're support
            if len(response_to) > 0:
                hit_rate = sum(1 for t in response_to if t in inbound_tweet_ids) / len(
                    response_to
                )
                if hit_rate > 0.5 and count > 1000:
                    brand_authors.add(author_id)

    return brand_authors


def filter_brand_conversations(
    df: pd.DataFrame, brand_name: str = None
) -> pd.DataFrame:
    """
    Filter dataset to only include conversations involving a specific brand.

    A conversation involves the brand if:
    - The brand authored an outbound tweet in the thread, OR
    - A customer's inbound tweet was responded to by the brand

    Args:
        df: Full dataset
        brand_name: Brand to filter for (default from config)

    Returns:
        Filtered DataFrame with only brand-relevant tweets
    """
    brand_name = brand_name or BRAND
    console.print(f"[bold blue]Filtering for brand: @{brand_name}[/bold blue]")

    brand_authors = identify_brand_author_ids(df, brand_name)

    if not brand_authors:
        # More aggressive: search tweet text for brand mentions
        console.print("[yellow]Using text-search fallback for brand detection...[/yellow]")
        brand_mask = df["text"].str.contains(brand_name, case=False, na=False)
        brand_tweet_ids = set(df[brand_mask]["tweet_id"])
        # Get all tweets in threads containing brand mentions
        related_ids = set()
        for _, row in df[brand_mask].iterrows():
            related_ids.add(row["tweet_id"])
            if row["in_response_to_tweet_id"] and row["in_response_to_tweet_id"] != "nan":
                related_ids.add(row["in_response_to_tweet_id"])
            if row["response_tweet_id"]:
                for rid in str(row["response_tweet_id"]).split(","):
                    rid = rid.strip()
                    if rid and rid != "nan":
                        related_ids.add(rid)
        filtered = df[df["tweet_id"].isin(related_ids)]
    else:
        console.print(f"[green]✓ Found {len(brand_authors)} brand author ID(s)[/green]")

        # Get all tweets authored by the brand
        brand_tweets = df[df["author_id"].isin(brand_authors)]
        brand_tweet_ids = set(brand_tweets["tweet_id"])

        # Get all tweets that the brand responded to
        responded_to_ids = set()
        for _, row in brand_tweets.iterrows():
            if row["in_response_to_tweet_id"] and str(row["in_response_to_tweet_id"]) != "nan":
                responded_to_ids.add(row["in_response_to_tweet_id"])

        # Get all tweets that responded to brand tweets
        responses_to_brand = set()
        for _, row in df.iterrows():
            if str(row["in_response_to_tweet_id"]) in brand_tweet_ids:
                responses_to_brand.add(row["tweet_id"])

        # Union of all relevant tweet IDs
        relevant_ids = brand_tweet_ids | responded_to_ids | responses_to_brand
        filtered = df[df["tweet_id"].isin(relevant_ids)]

    console.print(f"[green]✓ Filtered to {len(filtered):,} tweets for @{brand_name}[/green]")

    # Mark which tweets are from the brand
    filtered = filtered.copy()
    if brand_authors:
        filtered["is_brand"] = filtered["author_id"].isin(brand_authors)
    else:
        filtered["is_brand"] = ~filtered["inbound"]

    return filtered


def compute_brand_stats(df: pd.DataFrame, brand_name: str = None) -> dict:
    """Compute and save summary statistics for the filtered brand data."""
    brand_name = brand_name or BRAND

    inbound = df[df["inbound"]]
    outbound = df[~df["inbound"]]

    stats = {
        "brand": brand_name,
        "total_tweets": len(df),
        "inbound_tweets": len(inbound),
        "outbound_tweets": len(outbound),
        "unique_customers": inbound["author_id"].nunique(),
        "date_range": {
            "start": str(df["created_at"].min()),
            "end": str(df["created_at"].max()),
        },
        "avg_tweet_length": {
            "inbound": round(inbound["text"].str.len().mean(), 1),
            "outbound": round(outbound["text"].str.len().mean(), 1),
        },
    }

    # Save stats
    with open(BRAND_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Display
    table = Table(title=f"@{brand_name} Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total tweets", f"{stats['total_tweets']:,}")
    table.add_row("Customer messages", f"{stats['inbound_tweets']:,}")
    table.add_row("Brand responses", f"{stats['outbound_tweets']:,}")
    table.add_row("Unique customers", f"{stats['unique_customers']:,}")
    table.add_row("Date range", f"{stats['date_range']['start'][:10]} → {stats['date_range']['end'][:10]}")
    table.add_row("Avg tweet length (customer)", f"{stats['avg_tweet_length']['inbound']} chars")
    table.add_row("Avg tweet length (brand)", f"{stats['avg_tweet_length']['outbound']} chars")
    console.print(table)

    return stats
