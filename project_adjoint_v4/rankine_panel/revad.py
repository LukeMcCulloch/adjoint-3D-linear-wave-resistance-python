# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:07:10 2026

@author: Luke McCulloch

Vendored copy of automatic-differentiation-schemes-in-python/src/revad.py.
This is now a FORK, not a live link -- changes to the sibling repo's
revad.py won't automatically show up here. Vendored so rankine_panel doesn't
need a sys.path reach-across to a sibling repo for a core dependency (see
the #todo left in project_adjoint_v3/gradient_checks/trace_hs_influence_revad.py).

Beyond the original (Node, +-*/,  sin/cos/exp/log/sqrt/atan/atan2,
backward/grad/jacobian), this copy adds:
  - Node.__abs__            -> plain float (branch-condition use only, see below)
  - Node comparison dunders  -> plain bool (branch-condition use only)
  - sqrt/log/atan/atan2 now dispatch: Node in -> Node out (AD path),
    plain float in -> plain float out via math/np (fast path)

The dispatch additions are what let a single function (e.g.
rankine_panel/influence.py::hs_influence) serve BOTH the ordinary numeric
path and the differentiated path -- see that file for the actual payoff.

Why __abs__ and comparisons return plain values, not new Nodes: they're
only ever used for CONTROL FLOW (branch/guard conditions like `if d <= eps`),
never to compute a differentiable output. abs()/comparisons aren't smoothly
differentiable at their kink points anyway, and "branch on the primal value"
is the same convention used throughout this project's numba-facing AD work.
"""

# revad.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union
import math
import numpy as np

Array = np.ndarray


def _as1d(x) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("expected 1D array")
    return x


@dataclass
class Node:
    """
    Reverse-mode node (scalar-valued).
    Stores:
      - val: scalar numeric value
      - parents: list of (parent_node, local_derivative_wrt_parent)
      - grad: adjoint accumulated during backward pass
    """
    val: float
    parents: List[Tuple["Node", float]]
    grad: float = 0.0
    name: Optional[str] = None

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(float(other), [])
        out = Node(self.val + other.val, parents=[(self, 1.0), (other, 1.0)])
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(float(other), [])
        out = Node(self.val - other.val, parents=[(self, 1.0), (other, -1.0)])
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Node) else Node(float(other), [])
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(float(other), [])
        out = Node(self.val * other.val, parents=[(self, other.val), (other, self.val)])
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(float(other), [])
        if other.val == 0.0:
            raise ZeroDivisionError()
        out = Node(self.val / other.val,
                   parents=[(self, 1.0 / other.val),
                            (other, -self.val / (other.val * other.val))])
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Node) else Node(float(other), [])
        return other.__truediv__(self)

    def __neg__(self):
        return Node(-self.val, parents=[(self, -1.0)])

    def __pow__(self, p: float):
        # scalar power, p constant
        out = Node(self.val ** p, parents=[(self, p * (self.val ** (p - 1.0)))])
        return out

    def sqrt(self):
        """
        A plain method (not a dunder -- Python has no __sqrt__) named to
        match numpy's own object-array ufunc convention: np.sqrt(obj_array)
        looks for and calls each element's .sqrt() method when the dtype
        isn't a native numeric type. Confirmed empirically. Having this
        means np.sqrt works directly on arrays of Node -- no
        np.vectorize(sqrt) wrapper needed -- so
        rankine_panel/geometry.py::panel_geometry_all's two branches
        (ordinary float64 vs. AD) can share the literal same np.sqrt(...)
        call. The free sqrt(x) function below now just dispatches here.
        """
        s = float(np.sqrt(self.val))
        return Node(s, parents=[(self, 0.5 / s)])

    # ---- control-flow-only operators: plain values, no gradient tracking ----

    def __abs__(self):
        return abs(self.val)

    def _other_val(self, other):
        return other.val if isinstance(other, Node) else other

    def __lt__(self, other):
        return self.val < self._other_val(other)

    def __le__(self, other):
        return self.val <= self._other_val(other)

    def __gt__(self, other):
        return self.val > self._other_val(other)

    def __ge__(self, other):
        return self.val >= self._other_val(other)

    def __eq__(self, other):
        return self.val == self._other_val(other)

    def __hash__(self):
        return id(self)


def sin(x):
    if not isinstance(x, Node):
        return math.sin(x)
    return Node(float(np.sin(x.val)), parents=[(x, float(np.cos(x.val)))])


def cos(x):
    if not isinstance(x, Node):
        return math.cos(x)
    return Node(float(np.cos(x.val)), parents=[(x, float(-np.sin(x.val)))])


def exp(x):
    if not isinstance(x, Node):
        return math.exp(x)
    ex = float(np.exp(x.val))
    return Node(ex, parents=[(x, ex)])


def log(x):
    if not isinstance(x, Node):
        return math.log(x)
    return Node(float(np.log(x.val)), parents=[(x, 1.0 / x.val)])


def sqrt(x):
    if not isinstance(x, Node):
        return math.sqrt(x)
    return x.sqrt()


def atan(x):
    if not isinstance(x, Node):
        return math.atan(x)
    return Node(float(np.arctan(x.val)), parents=[(x, 1.0 / (1.0 + x.val * x.val))])


def atan2(y, x):
    """Quadrant-aware two-argument arctangent.

    d/dy atan2(y,x) = x/(x^2+y^2),  d/dx atan2(y,x) = -y/(x^2+y^2)
    """
    if not isinstance(y, Node) and not isinstance(x, Node):
        return math.atan2(y, x)
    y = y if isinstance(y, Node) else Node(float(y), [])
    x = x if isinstance(x, Node) else Node(float(x), [])
    denom = x.val * x.val + y.val * y.val
    val = float(np.arctan2(y.val, x.val))
    return Node(val, parents=[(y, x.val / denom), (x, -y.val / denom)])


def _toposort(root: Node) -> List[Node]:
    order: List[Node] = []
    seen = set()

    def dfs(n: Node):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p, _ in n.parents:
            dfs(p)
        order.append(n)

    dfs(root)
    return order


def backward(root: Node):
    # reset grads
    nodes = _toposort(root)
    for n in nodes:
        n.grad = 0.0
    root.grad = 1.0

    # reverse accumulation
    for n in reversed(nodes):
        g = n.grad
        for p, local in n.parents:
            p.grad += g * local


def grad(fun: Callable[[Sequence[Node]], Node], x: Array) -> Array:
    """
    Gradient of scalar function f: R^n -> R (Node scalar output).
    """
    x = _as1d(x)
    xs = [Node(float(xi), [], name=f"x{i}") for i, xi in enumerate(x)]
    y = fun(xs)
    backward(y)
    return np.array([v.grad for v in xs], dtype=np.float64)


def jacobian(fun: Callable[[Sequence[Node]], Sequence[Node]], x: Array) -> Array:
    """
    Jacobian of vector function F: R^n -> R^m using reverse-mode per output component.
    """
    x = _as1d(x)
    xs = [Node(float(xi), [], name=f"x{i}") for i, xi in enumerate(x)]  # anything coming in the x array is turned into a node! (with no parents i.e. it is an independent variable)
    ys = list(fun(xs))  # run the callable function, fun on the list of nodes just created from the x array supplied
    m = len(ys)
    n = len(xs)
    J = np.zeros((m, n), dtype=np.float64)
    for k, y in enumerate(ys):
        backward(y)
        J[k, :] = [v.grad for v in xs]
    return J
