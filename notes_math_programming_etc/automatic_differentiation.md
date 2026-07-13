
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


## terminology

Leaves = the independent variables. In jacobian(), these are the xs — Node objects built with parents=[] (empty list, literally nothing upstream). Nothing computed them; they're where the graph starts.

Root = whatever node you call backward() on. In jacobian()'s loop, each y in ys takes a turn being the root — root is just "the node I'm currently asking about," the top of whatever subgraph is reachable by walking backward through parents from that node.

"Seeded" means: the one line root.grad = 1.0. That's it — that's the entire seeding operation. Here's why it's 1.0 and why that's the right starting value.


Recall from the file itself: node.grad means "how much does the thing I called backward() on change, per unit change in this node." For root itself, that question is trivial: "how much does root change per unit change in root?" — obviously 1, since root is root. That's the base case the whole recursion bootstraps from. Everything else's .grad gets built up from that starting value as backward() walks outward (toward the leaves), via the chain-rule line p.grad += g * local.

Concretely, with the file's own f = x*x + x example (x=3): calling backward(f) does f.grad = 1.0 first. Then walking backward: g.grad picks up f.grad * 1.0 = 1.0 (from f's first parent-edge), and x.grad picks up contributions from both f directly (f.grad * 1.0) and from g (g.grad * 3.0 and g.grad * 3.0 again, once per occurrence of x in x*x) — ending at x.grad = 1 + 3 + 3 = 7 = 2(3)+1, matching the closed form. None of that would mean anything without first fixing f.grad = 1.0 as the anchor value everything else is computed relative to.

Two connections worth making explicit:

- This is the same "seed" word as in forward-mode AD, just planted at the opposite end — forward-mode seeds an input direction (asks "how much does everything move per unit move in this one input"), reverse-mode seeds the output (asks "how much does this one output move per unit move in everything upstream"). Same word, different end of the graph — the file's own __main__ docstring block says this explicitly, worth rereading now that it's concrete.

- It's also exactly the one-hot idea from a moment ago, just at its smallest possible scale: a scalar root has only one component to seed, so root.grad=1.0 is trivially "one-hot" — there's nothing else it could be zero relative to. jacobian()'s loop over ys is what turns this into genuine one-hot-over-many-outputs behavior: each backward(y) call first zeroes every reachable node's grad (including ones shared with other outputs' subgraphs — center, coordsys, cornerslocal, area all reuse a lot of the same upstream nodes like iv, jv, xic), then seeds just that one y to 1.0. So each of the 21 calls starts from a clean slate, seeds exactly one output, and nothing leaks between them even though they share most of the underlying graph.

---


---

# Performance

Q. Is reverse mode still better when there are many outputs too?
- You've found the actual sharp edge, and no — reverse mode is not automatically better once outputs get numerous. Let's get the real cost comparison right, because "one pass vs. many passes" is the wrong way to compare them.

Forward mode's real cost: one pass, but each elementary operation does $O(N)$ work (since every node carries a length-$N$ vector, and updating it — e.g. z.grad = y.val*x.grad_vec + x.val*y.grad_vec — touches all $N$ components). Total: $O(N \times \text{graph size})$.

Reverse mode's real cost: $m$ passes (one per output), but each pass does $O(1)$ work per elementary operation (pure scalars). Total: $O(m \times \text{graph size})$.

So the honest comparison is $N$ vs. $m$, not "1 pass" vs. "$m$ passes." Reverse mode wins exactly when $m < N$.

Check it against our actual kernels: hs_influence has $N=23$ inputs, $m=3$ outputs → reverse costs $3\times$graph-size vs. forward's $23\times$ — reverse wins by ~7.7×. phixx_influence has $N=23$, $m=9$ → reverse costs $9\times$ vs. $23\times$ — still wins, by ~2.6×, smaller margin but real. So for both kernels we actually built, there's no misunderstanding — reverse mode genuinely is cheaper, the numbers just work out that way because $m<N$ in both cases.

But you've spotted the real problem, and it's not a forward-vs-reverse question — it's what happens if you literally tried to differentiate the whole matrix A: $N\approx 10{,}665$ (all vertex components), and if you wanted the full dA/d(vertices) tensor, $m \approx N^2 \approx 11{,}300{,}000$ (every entry of A). Forward mode costs $\sim N\times(\text{graph size})$ — astronomically expensive because $N$ is huge already. Reverse mode costs $\sim m\times(\text{graph size})$ — also astronomically expensive, from the opposite direction, because $m$ is now huge. Neither mode wins — both lose, for different reasons. This isn't a "pick the smarter algorithm" situation; it's a "you're asking for an object that's too big to exist" situation.

That's exactly why the matrix-free principle from a few turns back isn't a performance optimization bolted on top of AD — it's the thing that makes reverse mode applicable at all here. phi := lam^T(A@sigma - b) collapses $m$ from 11 million down to 1 (one scalar), by contracting with lam and sigma before differentiating, not after. Once $m=1$, reverse mode is unconditionally the right choice regardless of how large $N$ (vertex count) is — that's the $m<N$ inequality satisfied about as favorably as it can be. So: your instinct that "many outputs" is a real problem for reverse mode is completely correct — we just never actually hand reverse mode a many-output problem. We always reduce to one scalar first.

(One honest caveat for later, not now: on real GPU hardware, forward mode's "$O(N)$ work per node, one pass" has a shape that maps very naturally onto wide SIMD/vector hardware, and reverse mode's "$m$ separate passes" can also be parallelized — run all $m$ backward sweeps concurrently across threads. So the FLOP-count argument above is the right first-order answer, but constant factors and parallelization opportunities matter more once you're actually on a GPU than they do in this asymptotic count. Worth remembering for the CUDA phase, not something to chase now.)




# Some details with backprop the first geometry layer 

The first layer is going from points and panels to all the local coordinate description.

And so in reverse mode, it goes from the local panel coordinates, area, etc., back to the points and panels description.



One-hot is standard terminology (borrowed from ML, but the concept predates that): a vector that's all zeros except a single entry set to 1 — the one "hot" (active) slot, everything else "cold" (zero). [0,0,1,0,0] is a one-hot vector with the hot slot at index 2.

And to your question directly: it's the opposite of what you said. The if o < 3: ... block doesn't build the gradient — it builds the query. panel_geometry_one_backward is where the actual gradient computation happens (every +=, every local-derivative multiply, all 21 reverse steps of chain rule). The seed-construction block just decides which question to hand it.

Here's why that distinction matters. panel_geometry_one_backward doesn't compute "the Jacobian" — it computes a vector-Jacobian product: given some weighting over the 21 outputs (how much you care about each one), it returns the weighted combination of that many rows of the Jacobian, summed together. Concretely, if $J$ is the true $(21,12)$ Jacobian and $s$ is your seed vector (length 21), the function computes $s^T J$ — a single length-12 vector, a weighted sum of rows.

If $s$ is one-hot — say 1 at slot o, zero everywhere else — then $s^T J$ degenerates to picking out row o alone, since every other row gets multiplied by 0 and vanishes from the sum. That's the whole trick: one-hot seeds are how you make a "weighted sum of rows" machine hand you back individual, unweighted rows, one at a time.

A tiny concrete example, separate from the panel code so the mechanism is crisp:

$$
g(x,y) = (xy, x+y),
\qquad
J =
\begin{pmatrix}
y & x \\
1 & 1
\end{pmatrix}.
$$

Suppose vjp(s0, s1) computes $s_0 \cdot(y,x) + s_1\cdot(1,1)$ (the weighted row-sum, exactly what a backward pass does for a 2-output function).

vjp(1, 0) → $(y, x)$ — row 0 alone. One-hot on output 0.
vjp(0, 1) → $(1, 1)$ — row 1 alone. One-hot on output 1.
vjp(1, 1) → $(y+1,\ x+1)$ — not one-hot, this is a genuine sum of both rows, not either individually. Useful for other things (e.g. "total sensitivity if both outputs matter equally"), but not for reconstructing $J$ row by row.
Same story in the real code: panel_geometry_one_backward(cache, d_center, d_coordsys, d_cornerslocal, d_area) is a general-purpose vector-Jacobian-product machine — it accepts any weighting across the 21 outputs (that's literally what production use looks like, where the real seed comes from lam, not a one-hot). The if o<3:... block is just the validation harness abusing that generality 21 times in a row, each time with a one-hot seed, purely to reconstruct the full matrix for comparison against the oracle. In actual production use (the eventual phi = lam^T(A@sigma-b) target), the seed won't be one-hot at all — it'll be whatever lam actually is, and you'll call the backward pass once, not 21 times, because you only ever want one scalar's gradient, not a full Jacobian.



