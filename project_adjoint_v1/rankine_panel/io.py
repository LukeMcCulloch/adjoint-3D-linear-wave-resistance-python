# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:31:06 2026

@author: tluke
"""

from __future__ import annotations
import numpy as np
from .types import PanelFile


def read_panel_file(path: str) -> PanelFile:
    """
    Faithful reader for my Fortran list-directed format:

      title (one line)
      i j
      npanels nfspanels npoints
      deltax
      panels(4, N)  (Fortran fills column-major)
      points(3, npoints) (Fortran fills column-major)

    Notes:
      - input panel indices in file are 1-based; returned panels are 0-based.
      - we reshape with order='F' to match Fortran's column-major fills.
    """
    with open(path, "r") as f:
        title = f.readline().rstrip("\n")
        tokens = f.read().split()

    it = iter(tokens)

    i = int(next(it)); j = int(next(it))
    npanels = int(next(it)); nfspanels = int(next(it)); npoints = int(next(it))
    deltax = float(next(it))

    N = npanels + nfspanels

    # panels: 4 x N
    panels_flat = [int(next(it)) for _ in range(4 * N)]
    panels = np.array(panels_flat, dtype=np.int64).reshape((4, N), order="F")
    panels -= 1  # 1-based -> 0-based

    # points: 3 x npoints
    points_flat = [float(next(it)) for _ in range(3 * npoints)]
    points = np.array(points_flat, dtype=np.float64).reshape((3, npoints), order="F")

    return PanelFile(title, i, j, npanels, nfspanels, npoints, deltax, panels, points)