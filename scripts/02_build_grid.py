"""Phase 2, steps 2.1-2.2: define the 1 km grid and assign cells to municipalities."""
from __future__ import annotations

import geopandas as gpd

from gisecon.config import data_path, ensure_dirs, load_config
from gisecon.data.grid import assign_cells, make_grid, write_template


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg"))
    assert munis.crs.to_string() == cfg["grid"]["crs"]

    # 2.1 grid template
    grid = make_grid(munis.total_bounds, cfg["grid"]["crs"], cfg["grid"]["cell_size_m"])
    grid.save(data_path(cfg, "interim", "grid.json"))
    write_template(grid, data_path(cfg, "interim", "grid_template.tif"))
    print(f"grid {grid.width} x {grid.height} = {grid.n_cells:,} cells, origin ({grid.xmin}, {grid.ymax})")

    # 2.2 cell -> municipality
    cells, rescued = assign_cells(grid, munis)
    n = cfg["study"]["n_municipalities"]
    got = cells["muni_code"].nunique()
    assert got == n, f"{got} municipalities have cells, expected {n}"
    cells.to_parquet(data_path(cfg, "interim", "cell_muni.parquet"), index=False)
    rescued.to_csv(data_path(cfg, "interim", "rescued_municipalities.csv"), index=False)

    per_muni = cells.groupby("muni_code").size()
    print(f"{len(cells):,} cells in the state; cells per municipality: "
          f"min {per_muni.min()}, median {per_muni.median():.0f}, max {per_muni.max()}")
    print(f"{len(rescued)} municipalities had no cell centre and were given their largest-overlap cell")
    if len(rescued):
        print(rescued.to_string(index=False))
    area = munis.geometry.area.sum() / 1e6
    print(f"state area {area:,.0f} km2 vs {len(cells):,} cells of 1 km2 (ratio {len(cells) / area:.4f})")


if __name__ == "__main__":
    main()
