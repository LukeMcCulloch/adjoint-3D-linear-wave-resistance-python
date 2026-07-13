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
    parents: List[Tuple["Node", float]]  # a node.parents = [(x,4.0), (y,3.0)] says that "one of my parents is node x; my local sesititity to that parent is 5.0"
    grad: float = 0.0 # note!: this is not the grad of this node.  
    # This grad says:
    # if this node wiggles a little, 
    # how much does the thing I called backward() on (the root)
    # wiggle in response.
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
        #
        # let: x.val = 2.0, y.val = 5.0, f = x*y.
        #
        # out = Node(self.val * other.val, parents=[(self, other.val), (other, self.val)])
        #
        # Here self is x, other is y. Substituting the actual values:
        #
        # out = Node(2.0*5.0, parents=[(x, 5.0), (y, 2.0)])
        #
        # so  df/dx  = 5.0
        # and df/dy = 2.0
        #
        # parents means (parent, basically local D (of the return function) with respect to this parent)
        #
        # not this automatically catches values on the way
        # this is possibly to do with it being an efficient dynamic programming method.
        #
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
        return Node(s, parents=[(self, 0.5 / s)])# note!: in geometry.py, we continue using the np.vectorize(sqrt) because it turns out to be 25% faster

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
    """
    the procedure that fills every node's accumulator correctl
    
    node.grad is the accumulator

    """
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
    xs = [Node(float(xi), [], name=f"x{i}") for i, xi in enumerate(x)] # anything coming in the x array is turned into a node!
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


if __name__ == "__main__":
    # ==========================================================================
    # Run this file directly (F5 in Spyder) to get x, g, f, a, b, h, etc. sitting
    # in the console namespace afterward -- inspect them, change x.val and
    # rerun bits by hand, whatever. This is meant to be poked at, not just read.
    # ==========================================================================
    
    """
    intuition:
        in forward mode, you pick an input direction first, seed it, 
        and propagate forward — so by the time you reach the output, 
        every intermediate (including the final one) is carrying "how much do I move per unit move in that one input I picked." 
        The answer lives at the end of the computation, same direction as the computation itself. 
    
    meanwhile:
        In reverse mode, you pick an output first, seed it with 1.0, 
        and propagate backward — so by the time you reach any given input, 
        it's carrying "how much does that one output I picked move per unit move in me." 
        The answer lives at the inputs, opposite direction from the computation. 
        Same math, opposite bookkeeping direction — that's genuinely disorienting the first time, 
        not a sign you're missing something.
    """

    print("="*70)
    print("Example 1: f(x) = x*x + x, by hand, one Node at a time")
    print("="*70)

    x = Node(3.0, [])
    print(f"x   = Node(val={x.val}, parents={x.parents})")

    g = x * x
    print(f"g=x*x  -> Node(val={g.val}, parents={g.parents})")
    print("   ^ note: TWO entries in g.parents, both pointing at x -- one per")
    print("     occurrence of x in x*x. Reuse of a variable = multiple parent edges.")

    f = g + x
    print(f"f=g+x  -> Node(val={f.val}, parents={f.parents})")

    print("\nBefore backward(): every .grad is still 0.0 -- forward pass only so far.")
    print(f"  x.grad={x.grad}  g.grad={g.grad}  f.grad={f.grad}")

    backward(f)

    print("\nAfter backward(f):")
    print(f"  f.grad={f.grad}   (seeded to 1.0: df/df=1, the starting point)")
    print(f"  g.grad={g.grad}   (got ONE contribution, from f's first parent-edge)")
    print(f"  x.grad={x.grad}   (got THREE contributions: one from f directly,")
    print("                    two from g, since x appeared twice in x*x)")
    print(f"  check: d/dx(x*x+x) = 2x+1 = {2*3.0+1.0}  -- matches x.grad exactly, "
          "and we never wrote that formula anywhere.")

    print("\n" + "="*70)
    print("Example 2: two independent leaves, h(a,b) = a*b + a")
    print("="*70)

    a = Node(3.0, [], name="a")
    b = Node(4.0, [], name="b")
    h = a * b + a
    print(f"h.val = {h.val}")
    backward(h)
    print(f"a.grad = {a.grad}   (dh/da = b + 1 = {4.0+1.0})")
    print(f"b.grad = {b.grad}   (dh/db = a     = {3.0})")

    print("\n" + "="*70)
    print("Example 3: the grad() and jacobian() convenience wrappers")
    print("(these do exactly the 'make leaf Nodes, run fun, call backward()' dance")
    print(" above for you -- see their source just above this block)")
    print("="*70)

    g_ab = grad(lambda xs: xs[0]*xs[1] + xs[0], np.array([3.0, 4.0]))
    print(f"grad(a*b+a) at (3,4) = {g_ab}   (matches Example 2's [a.grad, b.grad])")

    def F(xs):
        a, b = xs
        return [a*b, a + b]  # vector-valued: F(a,b) = (a*b, a+b)

    J = jacobian(F, np.array([3.0, 4.0]))
    print(f"jacobian(F) at (3,4) =\n{J}")
    print("  row 0 = d(a*b)/d(a,b) = (b, a) = (4, 3)")
    print("  row 1 = d(a+b)/d(a,b) = (1, 1)")

    print("\nTry in the console: build your own Node, do some arithmetic on it,")
    print("print .parents before backward(), then call backward(...) and look")
    print("at .grad. Try reusing a variable 3+ times and predict the number of")
    print("parent-edges before you count them.")
