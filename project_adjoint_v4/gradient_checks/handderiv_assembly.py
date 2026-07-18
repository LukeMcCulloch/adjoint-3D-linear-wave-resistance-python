# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for the FULL
assembly-level target -- the actual production goal of this whole series,
not "one panel/panel-pair at a time" anymore:

    phi(vertices) := lam^T (A(vertices) @ sigma - b(vertices))
                   = sum_row sum_col lam[row]*sigma[col]*A[row,col]
                     - sum_row lam[row]*b[row]

with lam (adjoint solution) and sigma (source strengths) held fixed --
already known from the forward+adjoint solves, never differentiated
through. See the plan's "IMPORTANT -- matrix-free design requirement"
section for why THIS is the target, not d(A)/d(vertices) itself (a
(N,N,3*Nverts) object, never formable).

WHAT'S NEW HERE, vs. every other handderiv_*.py file so far: everything up
to this point differentiated ONE panel or ONE panel-PAIR in isolation.
This file is the first one that actually SUMS over an assembly (multiple
rows and cols) and does the SCATTER-ADD -- taking each panel-pair's local
corner-gradients (4 corners x 3 coords each, in that panel's own private
indexing) and adding them into a GLOBAL, shared, per-VERTEX gradient
array, using the real integer panel-to-vertex connectivity
(`pf.panels[:, p]`). This is the piece flagged as "still-pending" in every
earlier file's docstring (see e.g. handderiv_panel_geometry.py's
docstring: "Nothing here does the corresponding SCATTER-ADD... needed once
this gets embedded in the full assembly loop").

FOUR CELL TYPES, each already validated as its own piece:
  - hull row, col==row      -> self_term_A_forward/backward
  - hull row, col!=row      -> A_entry_forward/backward (generic over hull/FS col)
  - FS row, any col         -> A_entry_fs_row_forward/backward (generic over hull/FS col)
  - b[row]: hull row        -> b_row_forward/backward
            FS row          -> literal constant 0, not differentiated

VALIDATION SCOPE: the real mesh has N=3360 panels (288 hull + 3072 FS) --
a full N^2 loop in pure Python is far too slow to be interactive. Instead
this validates on a small REAL sub-mesh: hull panels 0-3 (confirmed
spatially adjacent -- panel 0 and panel 1 share vertices 1 and 34, so
vertex-SHARING genuinely gets exercised, not just assumed) plus FS panels
300-301, gise N_sub=6. The oracle is built by composing the
ALREADY-VALIDATED revad-traced pieces (A_entry_revad, A_entry_fs_row_revad,
self_term_A_revad, b_row_revad) the exact same way the hand-derived
version composes their _forward/_backward counterparts -- same
composition, Node-graph-traced instead of hand-backward'd, differentiated
in ONE grad() call over every unique vertex touched by the sub-mesh.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from rankine_panel.io import read_panel_file
from rankine_panel.revad import Node, grad

from handderiv_A_entry import A_entry_forward, A_entry_backward
from handderiv_A_entry_fs_row import A_entry_fs_row_forward, A_entry_fs_row_backward
from handderiv_self_term_and_b import (
    self_term_A_forward, self_term_A_backward, b_row_forward, b_row_backward,
)

from trace_full_chain_A_entry_revad import A_entry_revad
from trace_full_chain_fs_row_revad import A_entry_fs_row_revad
from trace_self_term_and_b_revad import self_term_A_revad, b_row_revad


# =============================================================================
# Forward pass: sum every cell's contribution into phi, caching everything
# each cell's backward pass will need.
# =============================================================================

def assemble_phi_forward(pf, panel_indices, npanels_sub, nfspanels_sub,
                          deltax, U2, gravity, vinf_x, lam, sigma):
    N_sub = npanels_sub + nfspanels_sub
    assert len(panel_indices) == N_sub

    panel_corners = []
    panel_vert_idx = []
    for p in panel_indices:
        vidx = pf.panels[:, p]
        xyz = pf.points[:, vidx].T.copy()
        panel_corners.append([xyz[k] for k in range(4)])
        panel_vert_idx.append(vidx)

    phi = 0.0
    b_caches = {}
    A_caches = {}

    for row in range(N_sub):
        is_hull_row = row < npanels_sub
        row_corners = panel_corners[row]

        if is_hull_row:
            b_val, cache_b = b_row_forward(row_corners, vinf_x)
            phi += -lam[row] * b_val
            b_caches[row] = cache_b
        # FS rows: b[row] is the literal constant 0 -- no term, no cache.

        for col in range(N_sub):
            col_corners = panel_corners[col]
            if is_hull_row:
                if row == col:
                    A_val, cache_A = self_term_A_forward(row_corners)
                else:
                    A_val, cache_A = A_entry_forward(row_corners, col_corners)
            else:
                A_val, cache_A = A_entry_fs_row_forward(row_corners, col_corners, deltax, U2, gravity)
            phi += lam[row] * sigma[col] * A_val
            A_caches[(row, col)] = cache_A

    cache = dict(panel_corners=panel_corners, panel_vert_idx=panel_vert_idx,
                 b_caches=b_caches, A_caches=A_caches,
                 npanels_sub=npanels_sub, nfspanels_sub=nfspanels_sub)
    return phi, cache


# =============================================================================
# Backward pass: walk every cell in reverse, seed with lam[row]*sigma[col]
# (or -lam[row] for the b terms), and SCATTER-ADD each cell's local corner
# gradients into a global per-vertex array via the real vertex indices.
# =============================================================================

def assemble_phi_backward(cache, lam, sigma):
    panel_vert_idx = cache['panel_vert_idx']
    b_caches = cache['b_caches']
    A_caches = cache['A_caches']
    npanels_sub = cache['npanels_sub']
    nfspanels_sub = cache['nfspanels_sub']
    N_sub = npanels_sub + nfspanels_sub

    all_vert_idx = np.unique(np.concatenate(panel_vert_idx))
    slot = {v: i for i, v in enumerate(all_vert_idx)}
    d_vertices = np.zeros((3, len(all_vert_idx)))

    def scatter(vidx, d_corners):
        # each of the panel's 4 corners is a SEPARATE global vertex; a
        # vertex shared by another panel gets a SEPARATE scatter call later
        # for that panel's own contribution -- this += is what sums them.
        for k in range(4):
            d_vertices[:, slot[vidx[k]]] += d_corners[k]

    for row in range(N_sub):
        is_hull_row = row < npanels_sub

        if is_hull_row:
            d_row = b_row_backward(b_caches[row], -lam[row])
            scatter(panel_vert_idx[row], d_row)

        for col in range(N_sub):
            seed = lam[row] * sigma[col]
            cache_A = A_caches[(row, col)]
            if is_hull_row:
                if row == col:
                    d_row = self_term_A_backward(cache_A, seed)
                    scatter(panel_vert_idx[row], d_row)   # self-term only ever touches row's own corners
                else:
                    d_row, d_col = A_entry_backward(cache_A, seed)
                    scatter(panel_vert_idx[row], d_row)
                    scatter(panel_vert_idx[col], d_col)
            else:
                d_row, d_col = A_entry_fs_row_backward(cache_A, seed)
                scatter(panel_vert_idx[row], d_row)
                scatter(panel_vert_idx[col], d_col)

    return d_vertices, all_vert_idx


# =============================================================================
# Oracle: the SAME composition, built from the already-validated revad-traced
# pieces instead of hand backward passes, differentiated in one grad() call.
# =============================================================================

def make_phi_revad_traced(pf, panel_indices, npanels_sub, nfspanels_sub,
                           deltax, U2, gravity, vinf_x, lam, sigma, all_vert_idx):
    slot = {v: i for i, v in enumerate(all_vert_idx)}
    N_sub = npanels_sub + nfspanels_sub

    def run_traced(xs_nodes):
        vert_nodes = {v: [xs_nodes[3*i], xs_nodes[3*i+1], xs_nodes[3*i+2]]
                      for i, v in enumerate(all_vert_idx)}
        panel_corners_nodes = []
        for p in panel_indices:
            vidx = pf.panels[:, p]
            panel_corners_nodes.append([vert_nodes[v] for v in vidx])

        phi = Node(0.0, [])
        for row in range(N_sub):
            is_hull_row = row < npanels_sub
            row_corners = panel_corners_nodes[row]

            if is_hull_row:
                b_val = b_row_revad(row_corners, vinf_x)
                phi = phi - lam[row] * b_val

            for col in range(N_sub):
                col_corners = panel_corners_nodes[col]
                if is_hull_row:
                    if row == col:
                        A_val = self_term_A_revad(row_corners)
                    else:
                        A_val = A_entry_revad(row_corners, col_corners)
                else:
                    A_val = A_entry_fs_row_revad(row_corners, col_corners, deltax, U2, gravity)
                phi = phi + lam[row] * sigma[col] * A_val
        return phi

    return run_traced


# =============================================================================
# Validation
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    print(f"real mesh: npanels={pf.npanels}, nfspanels={pf.nfspanels}, N={pf.npanels+pf.nfspanels}")

    npanels_sub, nfspanels_sub = 4, 2
    panel_indices = [0, 1, 2, 3, 300, 301]
    assert panel_indices[npanels_sub] >= pf.npanels, "FS panel indices must be >= pf.npanels"
    N_sub = npanels_sub + nfspanels_sub

    Fr, gravity, length = 0.3, 9.80665, 1.0
    vinf_x = Fr * np.sqrt(length * gravity)
    U2 = vinf_x * vinf_x
    deltax = pf.deltax

    lam = np.array([1.3, -0.7, 0.5, 2.1, -1.1, 0.9])
    sigma = np.array([0.4, 1.6, -0.3, 0.8, -2.0, 1.2])

    phi, cache = assemble_phi_forward(pf, panel_indices, npanels_sub, nfspanels_sub,
                                       deltax, U2, gravity, vinf_x, lam, sigma)
    print("phi (hand):", phi)

    d_vertices, all_vert_idx = assemble_phi_backward(cache, lam, sigma)
    print("vertices touched by this sub-mesh:", all_vert_idx)
    print("d_vertices shape:", d_vertices.shape)

    # ---- oracle ----
    run_traced = make_phi_revad_traced(pf, panel_indices, npanels_sub, nfspanels_sub,
                                        deltax, U2, gravity, vinf_x, lam, sigma, all_vert_idx)
    x0 = pf.points[:, all_vert_idx].T.ravel()  # flat, matches vert_nodes' packing order
    phi_check = run_traced([Node(float(xi), [], name=f"x{i}") for i, xi in enumerate(x0)])
    print("phi (oracle, primal check):", phi_check.val)
    print("phi match:", np.isclose(phi, phi_check.val))

    J_oracle_flat = grad(run_traced, x0)          # (3*Nverts_sub,)
    J_oracle = J_oracle_flat.reshape(len(all_vert_idx), 3).T  # (3, Nverts_sub)

    err = np.max(np.abs(d_vertices - J_oracle))
    print(f"\nmax |hand - revad oracle| over all {d_vertices.size} vertex-gradient entries: {err:.3e}")

    i_worst = np.unravel_index(np.argmax(np.abs(d_vertices - J_oracle)), d_vertices.shape)
    print(f"worst entry: axis={('x','y','z')[i_worst[0]]}  global_vertex={all_vert_idx[i_worst[1]]}"
          f"  hand={d_vertices[i_worst]:.8e}  oracle={J_oracle[i_worst]:.8e}")

    print("\nper-vertex gradient (hand):")
    for i, v in enumerate(all_vert_idx):
        print(f"  vertex {v:5d}: {d_vertices[:, i]}")
