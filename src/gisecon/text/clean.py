"""Caption cleaning passes 1-2 (PLAN 4.7-4.8): drop filler, then drop claims the cell's data contradicts."""
from __future__ import annotations

import re
from typing import Mapping

FILLER = [r"comprehensive view", r"overall scene", r"gives? a sense of", r"the image shows a satellite view",
          r"^this (satellite|aerial) image", r"bird'?s[- ]eye view", r"captured from above"]

# word pattern -> check on the cell's numbers; a sentence that names the thing fails if the check is False
CLAIMS = {
    "water": (r"rivers?|lakes?|ponds?|reservoirs?|streams?|water bod(y|ies)",
              lambda c: c.get("lc_water", 0) >= 0.01 or c.get("water_km", 0) > 0),
    "forest": (r"forests?|woodlands?|wooded|tree cover|dense trees",
               lambda c: c.get("lc_tree", 0) >= 0.10),
    "farm": (r"crops?|cropland|fields?|plantations?|pastures?|farmland|agricultur\w*",
             lambda c: c.get("lc_crop", 0) + c.get("lc_grass", 0) >= 0.10),
    "building": (r"buildings?|houses?|residential|urban|rooftops?|roofs?",
                 lambda c: c.get("lc_built", 0) >= 0.02 or _poi_total(c) >= 1),
    "industry": (r"industrial|factor(y|ies)|warehouses?|min(e|es|ing)|quarr(y|ies)",
                 lambda c: c.get("poi_industry", 0) >= 1 or c.get("lc_built", 0) >= 0.05),
}
# ponytail: sentences with a negation ("no river is visible") skip the claim check; a parser would do better
NEGATION = re.compile(r"\b(no|not|without|absence|lack|none)\b", re.I)
_FILLER_RE = re.compile("|".join(FILLER), re.I)
_CLAIM_RE = {k: re.compile(rf"\b({p})\b", re.I) for k, (p, _) in CLAIMS.items()}


def _poi_total(c: Mapping) -> float:
    return sum(v for k, v in c.items() if k.startswith("poi_"))


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def clean_caption(text: str, cell: Mapping) -> tuple[str, list[dict]]:
    """(cleaned text, one {sentence, drop_reason} per sentence; drop_reason None = kept)."""
    rows = []
    for s in split_sentences(text):
        reason = "filler" if _FILLER_RE.search(s) else None
        if reason is None and not NEGATION.search(s):
            reason = next((f"claim_{k}" for k, (_, ok) in CLAIMS.items() if _CLAIM_RE[k].search(s) and not ok(cell)),
                          None)
        rows.append({"sentence": s, "drop_reason": reason})
    return " ".join(r["sentence"] for r in rows if r["drop_reason"] is None), rows
