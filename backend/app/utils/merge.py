"""
Merges consecutive segments from the same identified speaker when the gap
between them is small — this is what prevents a mid-sentence pause from
being treated as a hard turn boundary. Two DIFFERENT speakers are never
merged, even with zero gap, unless both resolved to the exact same
speaker_id (or both are None/"Unknown speaker", which we also don't merge,
since two different unknown people shouldn't collapse into one).
"""

def merge_by_identity(segments: list[dict], max_gap: float = 1.5) -> list[dict]:
    if not segments:
        return []

    merged = [dict(segments[0])]

    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]

        same_known_speaker = (
            seg["speaker_id"] is not None
            and seg["speaker_id"] == last["speaker_id"]
        )

        if same_known_speaker and gap <= max_gap:
            last["end"] = max(last["end"], seg["end"])
            last["confidence"] = max(last["confidence"], seg["confidence"])
        else:
            merged.append(dict(seg))

    return merged