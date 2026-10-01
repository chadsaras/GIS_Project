"""Satellite embeddings + text + night lights for municipal GDP estimation."""
import os as _os
import sys as _sys

# GDAL/PROJ locate their data through these variables, which are only set when the
# conda env is activated. Point them at the env so scripts also work when run directly.
# Only when the folders exist: pip-installed rasterio ships its own copies.
_share = _os.path.join(_sys.prefix, "share")
for _var, _dir in (("PROJ_DATA", "proj"), ("PROJ_LIB", "proj"), ("GDAL_DATA", "gdal")):
    if _os.path.isdir(_os.path.join(_share, _dir)):
        _os.environ.setdefault(_var, _os.path.join(_share, _dir))
