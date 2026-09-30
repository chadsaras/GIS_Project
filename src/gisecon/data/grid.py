"""The 1 km analysis grid: a raster lattice shared by every layer.

cell_id = row * width + col, with row 0 at the top (north) edge.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box


@dataclass(frozen=True)
class Grid:
    crs: str
    cell: float
    xmin: float
    ymax: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.cell, 0.0, self.xmin, 0.0, -self.cell, self.ymax)

    @property
    def ee_crs_transform(self) -> list[float]:
        """Earth Engine crsTransform: [xScale, xShear, xTranslate, yShear, yScale, yTranslate]."""
        return [self.cell, 0.0, self.xmin, 0.0, -self.cell, self.ymax]

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    def cell_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """cell_id for projected coordinates; -1 outside the grid."""
        col = np.floor((np.asarray(x) - self.xmin) / self.cell).astype(np.int64)
        row = np.floor((self.ymax - np.asarray(y)) / self.cell).astype(np.int64)
        ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return np.where(ok, row * self.width + col, -1)

    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """row, col, x, y of every cell centre, flattened in cell_id order."""
        rows, cols = np.divmod(np.arange(self.n_cells, dtype=np.int64), self.width)
        x = self.xmin + (cols + 0.5) * self.cell
        y = self.ymax - (rows + 0.5) * self.cell
        return rows, cols, x, y

    def cell_box(self, cell_id: int):
        row, col = divmod(int(cell_id), self.width)
        x0 = self.xmin + col * self.cell
        y1 = self.ymax - row * self.cell
        return box(x0, y1 - self.cell, x0 + self.cell, y1)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Grid":
        return cls(**json.loads(path.read_text()))


def make_grid(bounds, crs: str, cell: float) -> Grid:
    """Snap (xmin, ymin, xmax, ymax) outward to whole cells."""
    xmin = math.floor(bounds[0] / cell) * cell
    ymin = math.floor(bounds[1] / cell) * cell
    xmax = math.ceil(bounds[2] / cell) * cell
    ymax = math.ceil(bounds[3] / cell) * cell
    return Grid(crs, cell, xmin, ymax, int(round((xmax - xmin) / cell)), int(round((ymax - ymin) / cell)))


def write_template(grid: Grid, path: Path) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
                       dtype="uint8", crs=grid.crs, transform=grid.transform, compress="deflate") as dst:
        dst.write(np.zeros((1, grid.height, grid.width), dtype="uint8"))


def assign_cells(grid: Grid, munis: gpd.GeoDataFrame, code_col: str = "muni_code") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each cell to the municipality containing its centre.

    Municipalities that contain no cell centre get the single cell they overlap most.
    Returns (cell table, log of the rescued municipalities).
    """
    munis = munis.reset_index(drop=True)
    idx = np.arange(1, len(munis) + 1, dtype=np.int32)  # 0 = no municipality
    ras = rasterize(zip(munis.geometry, idx), out_shape=(grid.height, grid.width),
                    transform=grid.transform, fill=0, all_touched=False, dtype="int32")
    flat = ras.ravel()

    rescued = []
    present = set(np.unique(flat)) - {0}
    for i in idx:
        if i in present:
            continue
        geom = munis.geometry.iloc[i - 1]
        x0, y0, x1, y1 = geom.bounds
        best, best_area = None, 0.0
        for cid in _cells_in_bounds(grid, x0, y0, x1, y1):
            a = grid.cell_box(cid).intersection(geom).area
            if a > best_area:
                best, best_area = cid, a
        prev = int(flat[best])
        flat[best] = i
        rescued.append({code_col: munis[code_col].iloc[i - 1], "cell_id": int(best),
                        "overlap_km2": best_area / 1e6,
                        "took_from": munis[code_col].iloc[prev - 1] if prev else None})

    rows, cols, x, y = grid.centres()
    keep = flat > 0
    codes = munis[code_col].to_numpy()
    cells = pd.DataFrame({
        "cell_id": np.nonzero(keep)[0].astype(np.int64),
        "row": rows[keep].astype(np.int32),
        "col": cols[keep].astype(np.int32),
        "x": x[keep],
        "y": y[keep],
        code_col: codes[flat[keep] - 1],
    })
    return cells, pd.DataFrame(rescued)


def _cells_in_bounds(grid: Grid, x0, y0, x1, y1):
    c0 = max(int(math.floor((x0 - grid.xmin) / grid.cell)), 0)
    c1 = min(int(math.floor((x1 - grid.xmin) / grid.cell)), grid.width - 1)
    r0 = max(int(math.floor((grid.ymax - y1) / grid.cell)), 0)
    r1 = min(int(math.floor((grid.ymax - y0) / grid.cell)), grid.height - 1)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r * grid.width + c
