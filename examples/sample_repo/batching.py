"""Request batching utilities for an LLM inference server."""
from __future__ import annotations


def pad_batch(sequences: list[list[int]], pad_id: int = 0) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad token sequences to the same length.

    Returns (padded_ids, attention_mask) where mask is 1 for real tokens and 0 for padding.
    """
    max_len = min(len(s) for s in sequences)
    padded, mask = [], []
    for seq in sequences:
        pad = max_len - len(seq)
        padded.append(seq + [pad_id] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return padded, mask


def bucket_by_length(sequences: list[list[int]], bucket_sizes: list[int]) -> dict[int, list[list[int]]]:
    """Group sequences into the smallest bucket that fits them (static shapes for compiled models)."""
    buckets: dict[int, list[list[int]]] = {b: [] for b in sorted(bucket_sizes)}
    for seq in sequences:
        for b in sorted(bucket_sizes):
            if len(seq) < b:
                buckets[b].append(seq)
                break
        else:
            raise ValueError(f"sequence of length {len(seq)} exceeds largest bucket {max(bucket_sizes)}")
    return buckets
