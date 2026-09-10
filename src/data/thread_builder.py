"""
Thread builder to reconstruct multi-turn customer support conversations.

Reconstructs graph of tweet parent-child links:
in_response_to_tweet_id and response_tweet_id -> Linear / structured conversation chains.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import pandas as pd
from rich.console import Console

from src.config import CONVERSATIONS_PATH

console = Console()


@dataclass
class Message:
    tweet_id: str
    author_id: str
    role: str  # 'customer' or 'brand'
    text: str
    created_at: str
    turn: int


@dataclass
class Conversation:
    conversation_id: str
    brand: str
    messages: List[Message]
    first_customer_message: str
    first_brand_reply: Optional[str]
    num_turns: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "messages": [asdict(m) for m in self.messages],
            "first_customer_message": self.first_customer_message,
            "first_brand_reply": self.first_brand_reply,
            "num_turns": self.num_turns,
        }


def reconstruct_threads(
    df: pd.DataFrame,
    brand_name: str,
    max_conversations: int = 5000,
) -> List[Conversation]:
    """
    Reconstruct multi-turn conversation threads from raw tweets.
    Traces parent-child tweet references to assemble chronological dialogue turns.
    """
    console.print("[bold blue]Reconstructing conversation threads...[/bold blue]")

    # Create fast lookup by tweet_id
    tweet_dict = {}
    for _, row in df.iterrows():
        tweet_id = str(row["tweet_id"])
        tweet_dict[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": str(row["author_id"]),
            "is_brand": bool(row.get("is_brand", not row["inbound"])),
            "text": str(row["text"]),
            "created_at": str(row["created_at"]),
            "parent_id": str(row["in_response_to_tweet_id"]) if pd.notna(row["in_response_to_tweet_id"]) else None,
            "response_ids": [r.strip() for r in str(row["response_tweet_id"]).split(",") if r.strip() and r.strip() != "nan"],
        }

    # Find roots: inbound customer tweets with no parent_id or parent not in dataset
    root_tweet_ids = []
    for tid, data in tweet_dict.items():
        if not data["is_brand"]:
            parent = data["parent_id"]
            if not parent or parent == "nan" or parent not in tweet_dict:
                # Potential root of a conversation
                root_tweet_ids.append(tid)

    console.print(f"[cyan]Found {len(root_tweet_ids):,} candidate root tweets[/cyan]")

    conversations: List[Conversation] = []
    seen_tweets = set()

    for root_id in root_tweet_ids:
        if len(conversations) >= max_conversations:
            break

        curr_id = root_id
        thread_messages = []
        turn = 0

        # Traverse the response chain
        visited = set()
        while curr_id and curr_id in tweet_dict and curr_id not in visited:
            visited.add(curr_id)
            seen_tweets.add(curr_id)
            node = tweet_dict[curr_id]
            role = "brand" if node["is_brand"] else "customer"

            thread_messages.append(
                Message(
                    tweet_id=node["tweet_id"],
                    author_id=node["author_id"],
                    role=role,
                    text=node["text"],
                    created_at=node["created_at"],
                    turn=turn,
                )
            )
            turn += 1

            # Follow first response that exists in our tweet dictionary
            next_id = None
            for child_id in node["response_ids"]:
                if child_id in tweet_dict and child_id not in visited:
                    next_id = child_id
                    break
            curr_id = next_id

        # Keep threads that contain at least 1 customer message and at least 1 brand reply
        roles = [m.role for m in thread_messages]
        if "customer" in roles and "brand" in roles:
            first_cust = next((m.text for m in thread_messages if m.role == "customer"), "")
            first_brand = next((m.text for m in thread_messages if m.role == "brand"), None)

            conv = Conversation(
                conversation_id=root_id,
                brand=brand_name,
                messages=thread_messages,
                first_customer_message=first_cust,
                first_brand_reply=first_brand,
                num_turns=len(thread_messages),
            )
            conversations.append(conv)

    console.print(f"[green]✓ Reconstructed {len(conversations):,} full customer-brand conversations[/green]")

    # Save to jsonl
    with open(CONVERSATIONS_PATH, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict()) + "\n")

    return conversations
