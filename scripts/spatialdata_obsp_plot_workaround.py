"""Work around spatialdata-plot + AnnData .obsp copy failure for large n_obs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import anndata as ad

__all__ = [
    "obsp_stripped_for_plot",
    "plot_sdata_points_datashader",
]


@contextmanager
def obsp_stripped_for_plot(table: ad.AnnData) -> Iterator[None]:
    """
    spatialdata_plot's ``filter_by_coordinate_system(..., filter_tables=True)`` does
    ``table[mask].copy()``. With large ``n_obs`` and sparse graphs in ``.obsp``, that copy can
    raise ``ValueError: could not convert integer scalar`` (SciPy sparse / AnnData views).

    Temporarily clear ``.obsp`` for plotting, then restore so downstream Squidpy steps still
    see ``spatial_connectivities`` / ``connectivities``.
    """
    keys = list(table.obsp.keys())
    backup = {k: table.obsp[k] for k in keys}
    for k in keys:
        del table.obsp[k]
    try:
        yield
    finally:
        for k in list(table.obsp.keys()):
            del table.obsp[k]
        for k, v in backup.items():
            table.obsp[k] = v


def plot_sdata_points_datashader(
    sdata,
    *,
    element: str = "cells",
    color: str,
    table_key: str = "table",
) -> None:
    """
    Call ``sdata.pl.render_points(..., method='datashader').pl.show()`` with ``obsp`` safely
    stripped for the duration of the plot (large-n AnnData / spatialdata_plot bug workaround).

    Prefer this over a raw ``render_points`` call when ``n_obs`` is ~50k+ and the table has
    Scanpy/Squidpy graphs in ``.obsp``.
    """
    import spatialdata_plot  # noqa: F401 — registers .pl on SpatialData

    t = sdata.tables[table_key]
    with obsp_stripped_for_plot(t):
        sdata.pl.render_points(element, color=color, method="datashader").pl.show()
