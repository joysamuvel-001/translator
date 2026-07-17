"""
Merges consecutive segments from the same diarized voice when the gap
between them is small — this is what prevents a mid-sentence pause from
being treated as a hard turn boundary. Segments pyannote assigned to
different speaker labels are never merged, regardless of gap or of which
enrolled speaker they were identified as.
"""

def merge_by_identity(segments: list[dict], max_gap: float = 1.5) -> list[dict]:
    if not segments:
        return []

    merged = [dict(segments[0])]

    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]

        # pyannote's label is the only authority on whether the voice changed.
        # speaker_id must never be enough on its own: a misidentification can
        # stamp two different people with the same enrolled id, and merging on
        # that would discard a speaker boundary pyannote got right.
        same_diarized_voice = (
            seg["diarized_label"] == last["diarized_label"]
        )

        if same_diarized_voice and gap <= max_gap:
            last["end"] = max(last["end"], seg["end"])
            last["confidence"] = max(last["confidence"], seg["confidence"])
        else:
            merged.append(dict(seg))

    return merged