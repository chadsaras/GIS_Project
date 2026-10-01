"""Template "fact sentences" from a cell's land cover and OSM numbers (PLAN 4.6).

Night lights are deliberately left out, so Stage A never sees the Stage B label.
"""
from __future__ import annotations

from typing import Mapping

LC_WORDS = {"lc_built": "built-up", "lc_tree": "trees", "lc_grass": "grassland", "lc_crop": "cropland",
            "lc_shrub": "shrubland", "lc_water": "water", "lc_bare": "bare ground", "lc_wetland": "wetland"}
POI_WORDS = {"poi_education": ("school", "schools"), "poi_health": ("health facility", "health facilities"),
             "poi_retail": ("shop", "shops"), "poi_food": ("restaurant or café", "restaurants or cafés"),
             "poi_finance": ("bank", "banks"), "poi_industry": ("industrial site", "industrial sites"),
             "poi_office": ("office", "offices"), "poi_public": ("public building", "public buildings"),
             "poi_fuel": ("fuel station", "fuel stations"), "poi_tourism": ("tourist site", "tourist sites")}
ROAD_WORDS = {"road_major_km": "major roads", "road_medium_km": "medium roads", "road_minor_km": "minor roads",
              "road_track_km": "tracks"}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fact_sentences(c: Mapping) -> str:
    """Four short sentences: land cover, amenities, roads, waterways."""
    shares = sorted(((c.get(k, 0.0), w) for k, w in LC_WORDS.items()), reverse=True)
    top = [(s, w) for s, w in shares if s >= 0.05][:3]
    if top:
        lead = f"Mostly {top[0][1]}" if top[0][0] >= 0.5 else "Mixed land cover"
        out = [f"{lead}: " + ", ".join(f"{round(100 * s)}% {w}" for s, w in top) + "."]
    else:
        out = ["Land cover is fragmented."]

    pois = [(int(c.get(k, 0)), words) for k, words in POI_WORDS.items() if c.get(k, 0) >= 1]
    if pois:
        verb = "is" if len(pois) == 1 and pois[0][0] == 1 else "are"
        out.append(_join([f"{n} {sing if n == 1 else plur}" for n, (sing, plur) in pois]) + f" {verb} mapped.")
    else:
        out.append("No amenities are mapped.")

    roads = [f"{c.get(k, 0.0):.1f} km of {w}" for k, w in ROAD_WORDS.items() if c.get(k, 0.0) >= 0.1]
    out.append(_join(roads) + "." if roads else "No mapped roads.")
    w = c.get("water_km", 0.0)
    out.append(f"{w:.1f} km of mapped rivers or streams." if w >= 0.1 else "No mapped rivers.")
    return " ".join(out)
