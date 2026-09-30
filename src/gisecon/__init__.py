"""Satellite embeddings + text + night lights for municipal GDP estimation."""
import os as _os
import sys as _sys

# GDAL/PROJ locate their data through these variables, which are only set when the
# conda env is activated. Point them at the env so scripts also work when run directly.
_share = _os.path.join(_sys.prefix, "share")
_os.environ.setdefault("PROJ_DATA", _os.path.join(_share, "proj"))
_os.environ.setdefault("PROJ_LIB", _os.path.join(_share, "proj"))
_os.environ.setdefault("GDAL_DATA", _os.path.join(_share, "gdal"))
