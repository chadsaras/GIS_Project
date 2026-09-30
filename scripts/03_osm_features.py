"""Phase 2, steps 2.7-2.9: OpenStreetMap POI counts and road/waterway lengths per cell."""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyrosm import OSM

from gisecon.config import data_path, ensure_dirs, load_config
from gisecon.data import osm as osmf
from gisecon.data.grid import Grid


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    raw = data_path(cfg, "raw", "osm")
    src = raw / Path(cfg["sources"]["osm_url"]).name
    assert src.exists(), f"download {cfg['sources']['osm_url']} to {src} first"

    # 2.7 cut the Sudeste extract to the state (+2 km so border cells are complete)
    state_pbf = raw / f"{cfg['study']['state_abbr'].lower()}_{src.name.split('-')[1]}"
    if not state_pbf.exists():
        munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg"))
        poly = gpd.GeoSeries([munis.union_all().buffer(2000)], crs=munis.crs).to_crs(4326)
        geojson = raw / "state_boundary_buffer.geojson"
        poly.to_file(geojson, driver="GeoJSON")
        t = time.time()
        osmf.clip_pbf(src, geojson, state_pbf)
        print(f"clipped to {state_pbf.name} ({state_pbf.stat().st_size / 1e6:.0f} MB, {time.time() - t:.0f} s)")

    o = OSM(str(state_pbf))

    # 2.8 POIs
    t = time.time()
    pois = o.get_pois(custom_filter={k: True for k in osmf.POI_KEYS})
    counts = osmf.poi_counts(pois, grid)
    counts.to_parquet(data_path(cfg, "interim", "poi_counts.parquet"), index=False)
    totals = counts.drop(columns="cell_id").sum().astype(int)
    print(f"POIs: {len(pois):,} features read, {int(totals.sum()):,} categorised, "
          f"{len(counts):,} cells with at least one ({time.time() - t:.0f} s)")
    print(totals.to_string())

    # 2.9 roads and waterways
    t = time.time()
    roads = o.get_data_by_custom_criteria(custom_filter={"highway": sum(osmf.ROAD_CLASSES.values(), [])},
                                          filter_type="keep", keep_nodes=False, keep_relations=False)
    water = o.get_data_by_custom_criteria(custom_filter={"waterway": osmf.WATERWAYS},
                                          filter_type="keep", keep_nodes=False, keep_relations=False)
    lengths = osmf.road_water_lengths(roads, water, grid)
    lengths.to_parquet(data_path(cfg, "interim", "road_water_km.parquet"), index=False)
    print(f"roads: {len(roads):,} ways, waterways: {len(water):,} ways, {len(lengths):,} cells ({time.time() - t:.0f} s)")
    print(lengths.drop(columns="cell_id").sum().round(0).to_string())


if __name__ == "__main__":
    main()
