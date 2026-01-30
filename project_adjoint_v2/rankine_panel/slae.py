# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 00:34:44 2026

@author: tluke
"""

# rankine_panel/simqit.py
from __future__ import annotations
import numpy as np

try:
    from numba import njit
    import math
    HAS_NUMBA = True
except Exception:
    njit = None
    HAS_NUMBA = False


def _require_numba():
    if not HAS_NUMBA:
        raise ImportError("Numba is required for SIMQIT. Install it in this environment.")


if HAS_NUMBA:

    @njit(cache=True)
    def simqit_nb(A, R, N, sing, av, dx, X, max_outer=100, max_inner=2000):
        """
        Numba port of SIMQIT.

        Parameters
        ----------
        A : (N, N+2) float64
            IMPORTANT: stored as A[col, row], with:
              - rows 0..N-1 used for the linear system
              - row N   used as scratch for delta-X  (Fortran N+1)
              - row N+1 used for equation numbers    (Fortran N+2)
        R : (N,) float64
            Right-hand side (will be modified in-place)
        X : (N,) float64
            Initial guess (modified in-place; returned)

        Returns
        -------
        iter_count : int
        """
        # Initialize equation numbers row (Fortran: A(IS, N+2) = IS+0.5)
        eq_row = N + 1
        dx_row = N

        for is_ in range(N):
            A[is_, eq_row] = (is_ + 1) + 0.5

        # Initial residual update: R = R - A^T X  (in this A[col,row] storage)
        for iz in range(N):            # row index
            acc = 0.0
            for is_ in range(N):       # col index
                acc += X[is_] * A[is_, iz]
            R[iz] = R[iz] - acc

        averh = av * 1.4
        iter_count = 1

        # Outer loop label "3"
        for _outer in range(max_outer):
            averh = averh / 1.4
            iter_count += 1

            # For all "columns" IZ = 1..N (in Fortran naming)
            # Here IZ acts like the pivot column index, and IZ1 scans rows
            for iz in range(N):
                # Search pivot element in this column (scan rows iz..N-1)
                bmax = 0.0
                izpiv = iz
                for iz1 in range(iz, N):
                    val = abs(A[iz, iz1])  # A[col=iz, row=iz1]
                    if val > bmax:
                        bmax = val
                        izpiv = iz1

                if bmax < sing:
                    # signal singular
                    return -1

                alimit = bmax * averh
                bmax = math.copysign(bmax, A[iz, izpiv])

                # Exchange pivot and IZ rows; normalize pivot row
                if izpiv == iz:
                    # divide row iz by pivot (across all columns)
                    for is_ in range(N):
                        A[is_, iz] = A[is_, iz] / bmax
                else:
                    # swap rows iz and izpiv across all columns; normalize new row iz
                    for is_ in range(N):
                        save = A[is_, izpiv]
                        A[is_, izpiv] = A[is_, iz]
                        A[is_, iz] = save / bmax

                # swap equation numbers (stored in eq_row, per column)
                save = A[izpiv, eq_row]
                A[izpiv, eq_row] = A[iz, eq_row]
                A[iz, eq_row] = save

                # swap R entries (row-wise) and normalize R[iz]
                save = R[izpiv]
                R[izpiv] = R[iz]
                R[iz] = save / bmax

                # Eliminate large coefficients in rows below
                for iz1 in range(iz + 1, N):
                    aa = -A[iz, iz1]  # coefficient in this column at row iz1
                    if abs(aa) > alimit:
                        for is_ in range(N):
                            A[is_, iz1] = A[is_, iz1] + aa * A[is_, iz]
                        R[iz1] = R[iz1] + aa * R[iz]

            # Inner iteration label "60"
            dxmaxv = 1.0e30
            for _inner in range(max_inner):
                dxmax = 0.0

                # back-sub like sweep: determine delta-X stored in A(IZ, N+1)
                for iz in range(N - 1, -1, -1):
                    tmp = R[iz]
                    for is_ in range(iz + 1, N):
                        tmp = tmp - A[is_, iz] * A[is_, dx_row]
                    A[iz, dx_row] = tmp

                    aabs = abs(tmp)
                    if aabs > dxmax:
                        dxmax = aabs

                    X[iz] = X[iz] + tmp

                if dxmax < dx:
                    return iter_count

                # Update R (absolute value vector)
                for iz in range(N):
                    acc = 0.0
                    for is_ in range(0, iz):
                        acc = acc - A[is_, dx_row] * A[is_, iz]
                    R[iz] = acc

                # If no convergence, do another elimination step with smaller AVERH
                if dxmax > 0.8 * dxmaxv:
                    break

                dxmaxv = dxmax

        # If we reach here, we didn't converge within max_outer/max_inner
        return -2


def solve_simqit(A_phys: np.ndarray, b: np.ndarray,
                sing: float, av: float, dx: float,
                x0: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """
    Wrapper: solve A^T sigma = b using SIMQIT exactly like your Fortran call:
      SIMQIT(TRANSPOSE(am), b, ...)

    Here A_phys is the (row,col) matrix you assembled in Python as A.
    We build A_sim with A_sim[col,row] = A_phys[row,col] = A_phys.T[col,row]
    and add two extra rows (N and N+1) as SIMQIT scratch.
    """
    _require_numba()

    A_phys = np.asarray(A_phys, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).copy()

    N = A_phys.shape[0]
    if A_phys.shape[1] != N:
        raise ValueError("A_phys must be square")

    # Build SIMQIT storage A(col,row) with 2 extra rows
    A_sim = np.zeros((N, N + 2), dtype=np.float64)
    A_sim[:, :N] = A_phys.T  # col,row layout

    X = np.zeros(N, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    it = simqit_nb(A_sim, b, N, sing, av, dx, X)
    if it < 0:
        raise RuntimeError(f"SIMQIT failed with code {it} (singular or no convergence).")
    return X, int(it)
