"""Download IBGE municipal data: boundaries, GDP (SIDRA 5938), population (SIDRA 4709), regions."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

SIDRA = "https://apisidra.ibge.gov.br/values"
TIMEOUT = 120


def fetch_sidra(table: int, variables: list[int], year: int, state_code: int) -> pd.DataFrame:
    """Municipal values (level n6) for one state and year, long format: code, variable, value."""
    v = ",".join(str(x) for x in variables)
    url = f"{SIDRA}/t/{table}/n6/in%20n3%20{state_code}/v/{v}/p/{year}"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    rows = r.json()[1:]  # first row is the header description
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "muni_code": df["D1C"].astype(int),
        "variable": df["D2C"].astype(int),
        # SIDRA marks missing / suppressed values with symbols such as "-", "...", "X"
        "value": pd.to_numeric(df["V"], errors="coerce"),
    })
    return out


def fetch_municipios(api_url: str) -> pd.DataFrame:
    """Municipality list with immediate and intermediate region codes."""
    r = requests.get(api_url, timeout=TIMEOUT)
    r.raise_for_status()
    recs = []
    for m in r.json():
        ri = m["regiao-imediata"]
        recs.append({
            "muni_code": int(m["id"]),
            "muni_name": m["nome"],
            "rgi_code": int(ri["id"]),
            "rgi_name": ri["nome"],
            "rgint_code": int(ri["regiao-intermediaria"]["id"]),
            "rgint_name": ri["regiao-intermediaria"]["nome"],
        })
    return pd.DataFrame(recs)


def fetch_boundaries(url: str, dest_dir: Path) -> Path:
    """Download and unzip the IBGE municipal boundary shapefile; return the .shp path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    shp = next(dest_dir.glob("*.shp"), None)
    if shp is None:
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(dest_dir)
        shp = next(dest_dir.glob("*.shp"))
    return shp


def load_boundaries(shp: Path, crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp)
    gdf = gdf.rename(columns={"CD_MUN": "muni_code", "NM_MUN": "muni_name_shp", "AREA_KM2": "area_km2"})
    gdf["muni_code"] = gdf["muni_code"].astype(int)
    gdf = gdf[["muni_code", "muni_name_shp", "area_km2", "geometry"]].to_crs(crs)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf
