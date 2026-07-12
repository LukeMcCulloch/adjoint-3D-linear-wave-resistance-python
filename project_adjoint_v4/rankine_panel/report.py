# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 23:13:15 2026

@author: tluke
"""

# rankine_panel/report.py
from __future__ import annotations
import numpy as np

def append_output_dat(path: str, iter_count: int, sourcesink: float, cw: float, profilewave: np.ndarray) -> None:
    """
    Mirrors your io.output() routine: appends results to a text file.
    profilewave: (2,128) where row0 = x, row1 = zeta
    """
    profilewave = np.asarray(profilewave, dtype=float)

    with open(path, "a", newline="\n") as f:
        f.write("\n")
        f.write(" Results From the panel-hull Linear Free Surface Panel Code \n")
        f.write("---------------------------------------------------------------\n\n")
        f.write("source sink summation over all panels\n")
        f.write(f"{sourcesink: .8e}\n\n")
        f.write("Wave Resistance\n")
        f.write(f"{cw: .8e}\n\n")
        f.write("iteration count from the solver\n")
        f.write(f"{iter_count:d}\n\n\n")
        f.write("now output the wave profile on the centerline + ship hull\n")
        # Fortran wrote '(2D16.8)' in column-major; we’ll write x zeta pairs linewise
        for k in range(profilewave.shape[1]):
            f.write(f"{profilewave[0, k]: .8e} {profilewave[1, k]: .8e}\n")
        f.write("----------------------------------------------------------------\n")
