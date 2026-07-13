
# Automatic Differentiation (AD)

The chain rule is the fundemental building block of AD (not the product rule - my brain loves to switch these.)

$$\mathcal{C}\left(hain\left(rule\right)\right) \neq \mathcal{P}\left(roduct\right)\left(rule\right)$$

## Gradients and the Chain Rule:



Any computation is a DAG (directed acyclic graph) of elementary operations: 
- at the bottom of the graph, you have leaves $v_1, \ldots$, 
- then intermediate nodes each computed from earlier ones, 
- ending at some output $v_n$. 

The multivariate chain rule, written out fully for a DAG (not just a chain), says:

$$
\frac{\partial v_n}{\partial v_1}
=
\sum_{\text{every path } P: v_1 \to v_n}
\prod_{(i \to j)\in P}
\frac{\partial v_j}{\partial v_i}
$$

In words: 
- to get the total derivative of the output with respect to some early variable, 
- sum over every path through the graph from that variable to the output, and 
- along each path multiply the local one-hop derivatives. (from i to j)
- This is not an AD trick — it's just what the chain rule says once you unroll it through a graph with branches.


### why is it this?

- The sum-of-products decomposition is fundamental to differentiation of composed multivariable computations, not just a convenient AD trick.  The deepest reason is that a derivative is a linear map describing first-order change.  Suppose a node z depends on several intermediate variables:

$$z = f\left(u_1,...,u_m\right)$$

A small perturbation produces

$$  dz = \frac{\partial f}{\partial u_1} du_1 + ... + \frac{\partial f}{\partial u_m} du_m $$

- The branches add because the derivative is linear in the input perturbation.  
- If several independent components of a perturbation arrive at a node, their first-order effects superpose.
- That addition is not specific to multiplication. It holds for every differentiable multivariable function.

Branches add because a combined perturbation is decomposed into separate perturbations, and the derivative respects that decomposition by linearity.

$$
\mathcal{D}f_{(u,v)}[h]
=
\frac{\partial f}{\partial u}h_u
+
\frac{\partial f}{\partial v}h_v.
$$



or, as I was trying to hack at: 
$$ df = dfdu $$
$$ D f = df du$$

But really its:

$$
\mathcal{D}f
=
f_u e^u+f_v e^v
$$
where

$$
f_u=\frac{\partial f}{\partial u},
\qquad
f_v=\frac{\partial f}{\partial v}.
$$

---
### One final take at linearity and why the sum over paths shows up:

Start from the ordinary multivariable chain rule (the one from second-semester calculus, not an AD-specific statement). Suppose $z$ depends on $x$ through two intermediate quantities:

$$
p = f(x), \qquad q = g(x), \qquad z = h(p,q).
$$


$$
\frac{dz}{dx}
=
\frac{\partial z}{\partial p}\frac{dp}{dx}
+
\frac{\partial z}{\partial q}\frac{dq}{dx}.
$$

Look at this closely: it has exactly two terms, one per route from $x$ to $z$ (the $p$-route and the $q$-route), and each term is a product of the local derivatives along that route. This is not a graph-theory trick — it's the literal, standard, textbook multivariable chain rule. The "sum over paths" comes from the sum in this formula (additivity — when $x$ reaches $z$ two different ways, the two effects add). The "product along a path" comes from the ordinary single-variable chain rule applied along each individual route ($\frac{d}{dx}f(g(x)) = f'(g(x))\cdot g'(x)$ — composing functions in sequence multiplies their derivatives). Neither the sum nor the product is about multiplication-the-operation being special; the product shows up because composition always multiplies derivatives, and the sum shows up because branching-then-recombining always adds contributions

Concrete check, so nothing is hand-wavy: let $p = x^2$, $q = 2x$, and $z = p + q$. In other words,

$$
z(x) = x^2 + 2x.
$$

Directly, by ordinary single-variable differentiation,

$$
\frac{dz}{dx} = 2x + 2.
$$

Now use the two-path formula:

$$
\frac{\partial z}{\partial p} = 1,
\qquad
\frac{dp}{dx} = 2x,
\qquad
\frac{\partial z}{\partial q} = 1,
\qquad
\frac{dq}{dx} = 2.
$$

Therefore,

(1)(2x) + (1)(2)

2x + 2.


It matches exactly. These are two completely different-looking calculations with the same answer because they are the same mathematical statement.

Now generalize: 
- an arbitrary DAG is just this same two-route diamond, nested arbitrarily deep and arbitrarily wide (nodes with more than 2 parents just mean more terms in the sum; longer chains before recombining just mean longer products). 
- The global statement I wrote earlier ("sum over every path from $v_1$ to $v_n$ of the product of local derivatives along that path") is what you get by applying this exact 2-term chain rule repeatedly, at every branch point, and fully expanding the result. 
    - backward() doesn't compute that global sum directly (that would be the exponential-blowup naive approach, your 2017 code's issue) 
    - it computes the same total by applying the local 2-or-3-term version of this rule once per node, walking backward, letting each node's finished adjoint get reused by everyone upstream of it instead of being re-derived. 
    - That reuse — not any new mathematical idea — is the entire efficiency trick.

---

## Forward Mode AD

### A nice way to put it (vectorized forward AD)
Before running forward mode, you have to decide which input(s) you're differentiating with respect to. If you want just $\partial(\text{everything})/\partial x$ and nothing else, that's "one chosen input" — one forward pass, each node carries one scalar. If you want the sensitivity to every input simultaneously (which is what you'd normally want, and what your x.grad=[1,0], y.grad=[0,1] scheme does), you seed every leaf with its own basis vector and each node carries a length-$N$ vector through the same single forward pass. Same total work either way — the "N passes of scalars" and "1 pass of N-vectors" versions do the identical arithmetic, just organized differently. "Chosen inputs" just meant "whichever input direction(s) you decided to seed before starting."



## Reverse Mode AD

Let's start with an example:

$$ x = 2.0 \text{\space} \text{\space} \text{\space} \text{and}\text{\space} \text{\space} \text{\space} y = 5.0  $$


$$ f = x*y = 10.0 $$

$$ \frac{df}{dx} = x = 5.0 \qquad \text{and} \qquad \frac{df}{dy} = x = 2.0 $$

So f.parents = [(x, 5.0), (y, 2.0)]. Read the tuple (x, 5.0) as: "one of my parents is the node x; my local sensitivity to that parent is 5.0." The 5.0 is not x's value (x.val is still 2.0, completely unchanged — x is the same object, referenced by pointer) — it's $\partial f/\partial x$, which for multiplication happens to equal the other variable's value, y.val=5.0. That's the product rule doing exactly what the product rule does ($\partial(xy)/\partial x = y$)



---

## Directed Acrylic Graphs

Parents and children — precise definitions
- parents: A node's parents are the nodes it was built from — the immediate inputs to the operation that created it. For f = x*y, f.parents contains x and y. 
Your intuition ("parents are the primitive variables") is right in this particular example only because f is one hop from the leaves. 
In general, "parent" means "one hop upstream," not "leaf" — e.g. for g=x*x; f=g+x, g's parents are x (twice); f's parents are g and x. g is not a leaf, 
but it's still f's parent. 
Leaves are the special case: a node whose parents list is empty (nobody built it — it's an independent variable).

- children: children is the reverse relation: node f is a child of x because f used x as an input. Notice something important about the actual data structure: Node only stores .parents, never .children. That's not an oversight — at the moment x is constructed, it genuinely cannot know who will use it later (that code hasn't run yet). It can know its own parents, though, since they must already exist before x can be built from them. This asymmetry is why the algorithm has to be structured the way it is (see the DAG-direction question below).

A Directed Acyclic Graph, by definition, has no cycles — and a basic theorem about DAGs is that they always admit at least one valid topological order: an ordering of all nodes such that every edge points from an earlier node to a later one.

 In our case, the direction isn't an arbitrary convention we're imposing — it's forced by ordinary program execution: you cannot write x*y before x and y already exist. The order in which nodes get constructed is automatically a valid topological order (parents-before-children), because it's physically impossible to reference something before it's built. That's why "who used v_i" (its children) is well-defined even though we never stored a children list: _toposort's DFS walks from the root through .parents and appends each node after visiting all its parents, which produces exactly the parents-first order; backward() then processes reversed(nodes) — children-first — which is precisely the requirement for reverse accumulation (a node's adjoint can't be finalized until everything that used it has already contributed).

---