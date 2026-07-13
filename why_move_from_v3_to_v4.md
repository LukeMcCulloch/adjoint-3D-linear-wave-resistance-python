

# why v4?

v3 was frozen after being copied to create v4, and all the consolidation edits (retiring the hand-duplicated _revad functions in favor of the generic hs_influence/phixx_influence/panel_geometry_all) landed only in project_adjoint_v4. v3 still has the original, separately-hand-written, unrolled-style oracle functions — including the # todo: bring revad over comment still sitting there, since that fix (vendoring revad.py) only happened in v4 too.

So v3 is your intact learning copy, exactly as intended. The hs_influence_revad I was descri

v3 used hand-duplicated _revad functions that were built explicitly for accelerators.

- numba cannot jit-compile numpy?
- cuda cannot handle dynamic dispatch for tape-like rev ad