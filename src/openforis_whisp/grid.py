"""
Region grid mode for Whisp (zk-Whisp fork, branch ``zk-grid``).

Instead of computing statistics per plot, the Whisp indicator layers are computed once per cell of a
fixed global grid (aligned to the origin, EPSG:4326). Any plot can then be assessed from the grid
without calling Earth Engine again, and without sending the plot geometry anywhere: only grid-aligned
windows (or a whole region) are requested from Earth Engine.

Design (see CoBEST docs ``Step30-A2``):

* Base raster: pixels of ``pixel_deg`` (default 0.0001 deg, ~11 m) aligned to multiples of
  ``pixel_deg`` from (0, 0). Every band of the Whisp image is "mask x pixel area" (m2), plus ``Area``
  (= pixel area), exactly as in :func:`openforis_whisp.datasets.combine_datasets`.
* A grid cell of ``cell_px`` x ``cell_px`` pixels stores, per layer, the covered fraction of the cell
  ``sum(layer) / sum(Area)`` in [0, 1].
* A plot is assessed from the cells whose centre lies inside the plot polygon (a rule that can be
  checked inside a zk circuit). Plot coverage of a layer = mean of the cell fractions (cells of one
  small region have practically equal area).
* The Whisp perennial-crop decision tree (:func:`openforis_whisp.risk.add_risk_pcrop_col`) is then
  applied unchanged.

Only the global (non-national) layers used for ``risk_pcrop`` are downloaded by default.
"""

from __future__ import annotations

import io
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

AREA_BAND = "Area"
PCROP_THEMES = ("treecover", "commodities", "disturbance_before", "disturbance_after")
# Whisp default thresholds (percent of plot area) for indicators 1..4, see risk.whisp_risk
PCROP_THRESHOLDS = {"treecover": 10, "commodities": 50, "disturbance_before": 50, "disturbance_after": 10}


def pcrop_layers(national_codes=None) -> dict[str, list[str]]:
    """Layers used for ``risk_pcrop``, grouped by theme (order = lookup table order)."""
    from importlib.resources import files

    lk = pd.read_csv(files("openforis_whisp.parameters") / "lookup_datasets.csv")
    lk = lk[lk.use_for_risk_pcrop == 1]
    codes = {c.upper() for c in (national_codes or [])}
    lk = lk[lk.ISO2_code.isna() | lk.ISO2_code.str.upper().isin(codes)]
    return {t: lk[lk.theme == t].name.tolist() for t in PCROP_THEMES}


# ---------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """Grid-aligned window in integer pixel coordinates (col from lon=0, row from lat=0, north up)."""

    col0: int
    row0: int  # row index of the TOP edge: lat_top = -row0 * pixel_deg ... (see bounds)
    ncols: int
    nrows: int

    def bounds(self, pixel_deg: float):
        west = self.col0 * pixel_deg
        north = -self.row0 * pixel_deg
        return west, north - self.nrows * pixel_deg, west + self.ncols * pixel_deg, north


def window_for_bbox(bbox, pixel_deg: float, snap_px: int, margin_px: int) -> Window:
    """Smallest window that contains ``bbox`` plus a margin, with edges snapped to ``snap_px``."""
    minx, miny, maxx, maxy = bbox
    c0 = math.floor(minx / pixel_deg) - margin_px
    c1 = math.ceil(maxx / pixel_deg) + margin_px
    r0 = math.floor(-maxy / pixel_deg) - margin_px  # rows grow southwards
    r1 = math.ceil(-miny / pixel_deg) + margin_px
    c0, r0 = (c0 // snap_px) * snap_px, (r0 // snap_px) * snap_px
    c1, r1 = -((-c1) // snap_px) * snap_px, -((-r1) // snap_px) * snap_px
    return Window(c0, r0, c1 - c0, r1 - r0)


# ---------------------------------------------------------------------------------------------
# Download (Earth Engine) -> per-pixel band sums
# ---------------------------------------------------------------------------------------------


@dataclass
class PixelBlock:
    window: Window
    bands: list[str]
    data: np.ndarray  # (nbands, nrows, ncols) float64, m2 per pixel (Area = pixel area)


def _download_window(image, bands, win: Window, pixel_deg: float, retries: int = 4) -> PixelBlock:
    import ee
    import rasterio
    import requests
    import time

    west, _, _, north = win.bounds(pixel_deg)
    params = {
        "bands": bands,
        "crs": "EPSG:4326",
        "crs_transform": [pixel_deg, 0, west, 0, -pixel_deg, north],
        "dimensions": f"{win.ncols}x{win.nrows}",
        "format": "GEO_TIFF",
    }
    last = None
    for attempt in range(retries):
        try:
            url = image.select(bands).unmask(0).toFloat().getDownloadURL(params)
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            content = r.content
            if content[:2] == b"PK":  # zipped
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    content = z.read(next(n for n in z.namelist() if n.endswith(".tif")))
            with rasterio.MemoryFile(content) as mf, mf.open() as ds:
                arr = ds.read().astype(np.float64)
                tr = ds.transform
            if arr.shape != (len(bands), win.nrows, win.ncols):
                raise ValueError(f"unexpected shape {arr.shape}")
            if abs(tr.c - west) > pixel_deg / 100 or abs(tr.f - north) > pixel_deg / 100:
                raise ValueError(f"misaligned transform {tr}")
            return PixelBlock(win, list(bands), np.nan_to_num(arr, nan=0.0))
        except (ee.EEException, Exception) as e:  # noqa: BLE001 - retried, then raised
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for {win}: {last}")


def download_pixel_blocks(windows, pixel_deg=0.0001, layers=None, image=None, max_workers=8):
    """Download Area + layers for each grid-aligned window. Returns list[PixelBlock]."""
    from openforis_whisp.datasets import combine_datasets

    if layers is None:
        layers = [n for ls in pcrop_layers().values() for n in ls]
    if image is None:
        image = combine_datasets(include_context_bands=False)
    bands = [AREA_BAND] + list(layers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda w: _download_window(image, bands, w, pixel_deg), windows))


# ---------------------------------------------------------------------------------------------
# Grid cells
# ---------------------------------------------------------------------------------------------


@dataclass
class Grid:
    """Cell fractions for a set of cells of one global grid."""

    pixel_deg: float
    cell_px: int
    layers: list[str]
    # cell index (col, row) -> fractions (nlayers,) in [0, 1]
    cells: dict = field(default_factory=dict)

    @property
    def cell_deg(self) -> float:
        return self.pixel_deg * self.cell_px

    def cell_center(self, col: int, row: int):
        return ((col + 0.5) * self.cell_deg, -(row + 0.5) * self.cell_deg)


def grid_from_blocks(blocks, cell_px: int, pixel_deg: float = 0.0001) -> Grid:
    """Aggregate pixel blocks into cells of ``cell_px`` pixels. Windows must be snapped to cell_px."""
    b0 = blocks[0]
    g = Grid(pixel_deg=pixel_deg, cell_px=cell_px, layers=b0.bands[1:])
    for b in blocks:
        w = b.window
        if w.col0 % cell_px or w.row0 % cell_px or w.ncols % cell_px or w.nrows % cell_px:
            raise ValueError(f"window {w} not aligned to cell_px={cell_px}")
        nb, nr, nc = b.data.shape
        s = b.data.reshape(nb, nr // cell_px, cell_px, nc // cell_px, cell_px).sum(axis=(2, 4))
        area = s[0]
        frac = np.divide(s[1:], area, out=np.zeros_like(s[1:]), where=area > 0).clip(0, 1)
        for i in range(nr // cell_px):
            for j in range(nc // cell_px):
                g.cells[(w.col0 // cell_px + j, w.row0 // cell_px + i)] = frac[:, i, j]
    return g


def quantize(frac: np.ndarray, bits: int | None) -> np.ndarray:
    """Quantise fractions to ``bits`` bits (round to nearest level); None = no quantisation."""
    if bits is None:
        return frac
    levels = (1 << bits) - 1
    return np.rint(frac * levels) / levels


# ---------------------------------------------------------------------------------------------
# Plot assessment from the grid
# ---------------------------------------------------------------------------------------------


def cells_in_polygon(grid: Grid, polygon):
    """Grid cells whose centre lies inside ``polygon`` (shapely, lon/lat)."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    minx, miny, maxx, maxy = polygon.bounds
    cd = grid.cell_deg
    p = prep(polygon)
    out = []
    for col in range(math.floor(minx / cd), math.ceil(maxx / cd) + 1):
        for row in range(math.floor(-maxy / cd), math.ceil(-miny / cd) + 1):
            if p.contains(Point(grid.cell_center(col, row))):
                out.append((col, row))
    return out


def assess_plots_from_grid(gdf, grid: Grid, bits: int | None = 4, mode: str = "per_layer",
                           id_column: str = "code", themes: dict | None = None,
                           thresholds: dict | None = None) -> pd.DataFrame:
    """
    Whisp ``risk_pcrop`` for each polygon computed only from the grid.

    mode:
      ``per_layer``: coverage per layer (mean of quantised cell fractions), indicator = any layer
                     of the theme above threshold (same rule as Whisp).
      ``per_theme``: one value per theme per cell = max over the theme's layers in that cell
                     (smaller commitment), indicator = mean of that value above threshold.
    """
    from openforis_whisp.risk import add_risk_pcrop_col

    themes = themes or pcrop_layers()
    thresholds = thresholds or PCROP_THRESHOLDS
    idx = {n: i for i, n in enumerate(grid.layers)}
    names = {"treecover": "Ind_01_treecover", "commodities": "Ind_02_commodities",
             "disturbance_before": "Ind_03_disturbance_before_2020",
             "disturbance_after": "Ind_04_disturbance_after_2020"}
    rows = []
    for _, r in gdf.iterrows():
        cells = cells_in_polygon(grid, r.geometry)
        rec = {id_column: r[id_column], "n_cells": len(cells)}
        if not cells:
            rows.append(rec)
            continue
        missing = [c for c in cells if c not in grid.cells]
        if missing:
            raise KeyError(f"{r[id_column]}: {len(missing)} cells not in grid")
        F = quantize(np.stack([grid.cells[c] for c in cells]), bits)  # (ncells, nlayers)
        for t, layers in themes.items():
            li = [idx[n] for n in layers if n in idx]
            if mode == "per_layer":
                pct = 100 * F[:, li].mean(axis=0)
                val = float(pct.max()) if len(li) else 0.0
            elif mode == "per_theme":
                val = float(100 * quantize(F[:, li].max(axis=1), bits).mean()) if len(li) else 0.0
            else:
                raise ValueError(mode)
            rec[f"{t}_pct"] = val
            rec[names[t]] = "yes" if val > thresholds[t] else "no"
        rows.append(rec)
    df = pd.DataFrame(rows)
    ok = df.n_cells > 0
    df.loc[ok, "risk_pcrop"] = add_risk_pcrop_col(
        df.loc[ok].copy(), *names.values()
    )["risk_pcrop"].values
    df.loc[~ok, "risk_pcrop"] = "no_cells"
    return df
