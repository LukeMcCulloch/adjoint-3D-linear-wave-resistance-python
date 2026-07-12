# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:30:07 2026

@author: Luke McCulloch
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class PanelFile:
    title: str
    i: int
    j: int
    npanels: int
    nfspanels: int
    npoints: int
    deltax: float
    panels: Array   # (4, N) int64, 0-based
    points: Array   # (3, npoints) float64

    @property
    def N(self) -> int:
        return self.npanels + self.nfspanels


@dataclass(frozen=True)
class PanelGeometry:
    center: Array        # (3, N)
    coordsys: Array      # (3,3,N) columns: t1,t2,n
    cornerslocal: Array  # (2,4,N)
    area: Array          # (N,)