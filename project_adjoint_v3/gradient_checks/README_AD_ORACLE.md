# Reverse-mode AD oracle — file map

Quick reference for mapping the math (`d(what)/d(what)`) onto the actual
Python files/functions. All of these are **oracles**: pure Python, build a
full `Node` object graph via operator overloading (from `revad.py` in the
sibling repo `automatic-differentiation-schemes-in-python`), correctness
only, never meant to be fast. Every file's `if __name__ == "__main__":`
block cross-checks its Jacobian against central-FD on real `fifi.dat` panel
data and prints max absolute/relative error.

This is **tracing**, not AST/source transformation: we call each function
with `Node` objects instead of floats and let Python's own operator
dispatch (`__add__`, `__mul__`, ...) build the graph as a side effect of
normal execution. No parsing of source code happens anywhere here.

## Dependency chain

```
revad.py (sibling repo)
    Node, atan, atan2, sqrt, log, jacobian()  <- the reverse-mode engine itself

trace_panel_geometry_revad.py
    panel_geometry_revad()          d(panel geometry)/d(vertices)
        |
        +-----------------------+
        v                       v
trace_hs_influence_revad.py    trace_phixx_influence_revad.py
    hs_influence_revad()           phixx_influence_revad()
    d(velocity)/d(geometry)        d(Hessian)/d(geometry)
        |                       |
        +-----------+-----------+
                     v
   trace_full_chain_revad.py         trace_full_chain_phixx_revad.py
       combined_revad()                  combined_phixx_revad()
       d(velocity)/d(vertices)           d(Hessian)/d(vertices)
                     |
        +------------+------------+
        v                         v
trace_full_chain_A_entry_revad.py   trace_full_chain_fs_row_revad.py
    A_entry_revad()                     A_entry_fs_row_revad()
    d(A[row,col])/d(vertices),          d(A[row,col])/d(vertices),
    hull/body row                       free-surface row

trace_self_term_and_b_revad.py
    self_term_A_revad(), b_row_revad()
    (uses panel_geometry_revad only -- no kernel calls)
```

## File-by-file

### `trace_panel_geometry_revad.py`
**Computes:** `d(panel geometry)/d(vertices)` — one panel's 4 corner
vertices (12 scalars) → `center`, `coordsys`, `cornerslocal`, `area` (21
scalars).
**Function:** `panel_geometry_revad(corners)`. Node-for-node port of
`geometry.py::panel_geometry_all`, for a single panel.
**No composition** — this is the base of the chain, everything else builds
on it.

### `trace_hs_influence_revad.py`
**Computes:** `d(velocity)/d(panel geometry)` — `fieldpoint`, `center`,
`coordsys`, `corners_local` (23 scalars) → induced velocity (3 scalars).
**Function:** `hs_influence_revad(...)`. Port of `influence.py::hs_influence`
/ `numba_kernels.py::hs_influence_nb`.
**No composition** — takes panel geometry directly, doesn't know about
vertices at all.

### `trace_phixx_influence_revad.py`
**Computes:** `d(Hessian)/d(panel geometry)` — same 23 inputs, 3×3 Hessian
(9 scalars) out.
**Function:** `phixx_influence_revad(...)`. Deliberately ported from the
*production* `numba_kernels.py::phixx_influence_nb`, not the divergent
`influence.py::phixx_influence` (extra "double the first block" line there,
unresolved discrepancy, parked).

### `trace_full_chain_revad.py`
**Computes:** `d(velocity)/d(vertices)` — two panels' vertices (24 scalars:
"row" panel = field point, "col" panel = source) → velocity (3 scalars).
**Composition happens in `combined_revad(row_corners, col_corners)`:**
```python
center_row, _, _, _ = panel_geometry_revad(row_corners)              # row's own center -> field point
center_col, coordsys_col, cornerslocal_col, _ = panel_geometry_revad(col_corners)  # col's full geometry
vel = hs_influence_revad(center_row, center_col, coordsys_col, cornerslocal_col)
```
That's the whole composition — call geometry twice, feed the results into
the kernel. Reverse-mode composes automatically because everything's still
`Node` objects.

### `trace_full_chain_phixx_revad.py`
**Computes:** `d(Hessian)/d(vertices)`. Identical structure to the file
above, in `combined_phixx_revad(row_corners, col_corners)`, just calling
`phixx_influence_revad` instead of `hs_influence_revad` as the last step.

### `trace_full_chain_A_entry_revad.py`
**Computes:** `d(A[row,col])/d(vertices)` for a **hull/body row**.
**Composition happens in `A_entry_revad(row_corners, col_corners)`:**
```python
center_row, coordsys_row, ... = panel_geometry_revad(row_corners)   # need row's normal too, this time
center_col, coordsys_col, cornerslocal_col, ... = panel_geometry_revad(col_corners)
normal_row = coordsys_row[:, 2]                                     # 3rd column = normal
fieldpoint = center_row
fieldpoint_mirror = [center_row.x, -center_row.y, center_row.z]     # y-flip
v  = hs_influence_revad(fieldpoint,        center_col, coordsys_col, cornerslocal_col)
vp = hs_influence_revad(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
v_combined = [v.x+vp.x, v.y-vp.y, v.z+vp.z]                          # even,odd,even in y
A_entry = dot(normal_row, v_combined)
```
Two things this adds beyond the plain velocity chain: the **dot-with-normal**
(needs `coordsys_row`, which the velocity-only chain discards) and the
**mirror term** (calls the kernel twice, combines with even/odd signs).
Doesn't cover `row == col` — see self-term below.

### `trace_full_chain_fs_row_revad.py`
**Computes:** `d(A[row,col])/d(vertices)` for a **free-surface row**.
**Composition happens in `A_entry_fs_row_revad(row_corners, col_corners, deltax, U2, gravity)`:**
```python
center_row, ... = panel_geometry_revad(row_corners)
center_col, coordsys_col, cornerslocal_col, ... = panel_geometry_revad(col_corners)
p0 = center_row.x + deltax    # real geometry + constant offset
p1 = center_row.y             # real geometry, unmodified
p2 = Node(0.0, [])            # ** hard-set constant, NOT center_row.z ** -- see docstring
fieldpoint = [p0, p1, p2];  fieldpoint_mirror = [p0, -p1, Node(0.0, [])]
v  = hs_influence_revad(fieldpoint, ...);   vp  = hs_influence_revad(fieldpoint_mirror, ...)
H  = phixx_influence_revad(fieldpoint, ...); Hp = phixx_influence_revad(fieldpoint_mirror, ...)
A_entry = (H[0][0]+Hp[0][0])*U2 + (v[2]+vp[2])*gravity
```
The one genuinely non-obvious piece is the `z = 0` line: it's a disconnected
constant, not `center_row.z`, so the field point's z-derivative w.r.t.
vertices is identically zero (confirmed numerically in this file's sanity
check — both AD and FD independently print exactly `0.000e+00`).

### `trace_self_term_and_b_revad.py`
**Computes two small, unrelated things, both needing only
`panel_geometry_revad` — no kernel calls at all:**
- `self_term_A_revad(row_corners)`: the hull diagonal entry
  `A[row,row] = -0.5 * dot(normal_row, normal_row)`. Since `normal_row` is
  unit-length by construction, this is a **structural constant** (`-0.5`,
  always) — confirmed to have ~zero derivative (machine roundoff), not
  assumed.
- `b_row_revad(row_corners, vinf_x)`: `b[row] = normal_row.x * vinf_x`
  (`vinf_x` is a flow constant, not geometry). Genuinely nonzero derivative,
  validated against FD.

No composition beyond a single `panel_geometry_revad(row_corners)` call in
each.
