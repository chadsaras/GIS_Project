"""Caption cleaning passes 1-2 (PLAN 4.7-4.8): drop filler, then drop claims the cell's data contradicts."""
from __future__ import annotations

import re
from typing import Mapping

FILLER = [r"comprehensive view", r"overall (scene|impression|picture)", r"gives? a sense of",
          r"the image shows a satellite view", r"bird'?s[- ]eye view", r"captured from above", r"^overall,"]
# Note: sentences that merely *start* with "This satellite image shows ..." are kept: they usually carry the
# main description (e.g. "... captures a densely populated urban area").

# word pattern -> check on the cell's numbers; a sentence that names the thing fails if the check is False
CLAIMS = {
    "water": (r"rivers?|lakes?|ponds?|reservoirs?|streams?|water bod(y|ies)",
              lambda c: c.get("lc_water", 0) >= 0.01 or c.get("water_km", 0) > 0),
    "forest": (r"forests?|woodlands?|wooded|tree cover|dense trees",
               lambda c: c.get("lc_tree", 0) >= 0.10),
    "farm": (r"crops?|cropland|fields?|plantations?|pastures?|farmland|agricultur\w*",
             lambda c: c.get("lc_crop", 0) + c.get("lc_grass", 0) >= 0.10),
    # dense / urban claims need a real settlement; "any building" claims need just one detected building
    "urban": (r"densely (built|populated)|dense (urban|residential|housing)|urban|city|cities|town|downtown|"
              r"high-rise|apartment blocks?|multi-stor(e)?y",
              lambda c: c.get("lc_built", 0) >= 0.05 or _buildings(c) >= 50),
    "building": (r"buildings?|houses?|homes?|residential|rooftops?|roofs?|structures?|settlements?|villages?",
                 lambda c: c.get("lc_built", 0) >= 0.02 or _poi_total(c) >= 1 or _buildings(c) >= 1),
    # paved-road words only: dirt tracks and paths are often missing from OSM, so they are not checked
    "road": (r"roads?|streets?|highways?|avenues?",
             lambda c: _road_km(c) > 0 or c.get("lc_built", 0) >= 0.01 or _buildings(c) >= 5),
    "industry": (r"industrial|factor(y|ies)|warehouses?|min(e|es|ing)|quarr(y|ies)",
                 lambda c: c.get("poi_industry", 0) >= 1 or c.get("lc_built", 0) >= 0.05),
}
# Building evidence (scripts/03b_buildings.py): Microsoft ML footprints + OSM buildings, so isolated farm
# buildings that WorldCover's 10 m built-up class misses still count.
# ponytail: sentences with a negation ("no river is visible") skip the claim check; a parser would do better
NEGATION = re.compile(r"\b(no|not|without|absence|lack|none)\b", re.I)
_FILLER_RE = re.compile("|".join(FILLER), re.I)
_CLAIM_RE = {k: re.compile(rf"\b({p})\b", re.I) for k, (p, _) in CLAIMS.items()}


def _poi_total(c: Mapping) -> float:
    return sum(v for k, v in c.items() if k.startswith("poi_"))


def _buildings(c: Mapping) -> float:
    return max(c.get("ms_buildings", 0), c.get("osm_buildings", 0))


def _road_km(c: Mapping) -> float:
    return sum(v for k, v in c.items() if k.startswith("road_") and k.endswith("_km"))


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def clean_caption(text: str, cell: Mapping) -> tuple[str, list[dict]]:
    """(cleaned text, one {sentence, drop_reason} per sentence; drop_reason None = kept)."""
    rows = []
    sents = split_sentences(text)
    for i, s in enumerate(sents):
        if i == len(sents) - 1 and not s.rstrip().endswith((".", "!", "?")):  # cut off at the token limit
            rows.append({"sentence": s, "drop_reason": "truncated"})
            continue
        reason = "filler" if _FILLER_RE.search(s) else None
        if reason is None and not NEGATION.search(s):
            reason = next((f"claim_{k}" for k, (_, ok) in CLAIMS.items() if _CLAIM_RE[k].search(s) and not ok(cell)),
                          None)
        rows.append({"sentence": s, "drop_reason": reason})
    return " ".join(r["sentence"] for r in rows if r["drop_reason"] is None), rows
