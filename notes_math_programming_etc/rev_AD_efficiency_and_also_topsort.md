

# Efficiecny, Ordering (baby topology) , Dynamic Programming

 ### Topsort

- Why do we use topsort in reverse mode AD?


### The Topological Sort

Reverse-mode automatic differentiation uses a **topological ordering** because derivatives must be propagated backward through the computational graph only after all downstream contributions are known.

Suppose

$$
a = f(x), \qquad
b = g(a), \qquad
c = h(a), \qquad
y = k(b,c).
$$

The graph is

```text
x → a → b ─┐
      └→ c ─┴→ y
```


In reverse mode, we compute adjoints such as

$$
\bar{a} \equiv \frac{\partial y}{\partial a}.
$$

---
Note: $\bar{a}$ means "the adjoint of a"
Because \(a\) influences \(y\) through both \(b\) and \(c\),

$\bar{v}_i$ means “the adjoint of $v_i$,” defined as:

$$
\bar{v}_i
:=
\frac{
\partial \left(\text{whatever output we called `backward()` on}\right)
}{
\partial v_i
}.
$$

---

$$
\bar{a}
=
\bar{b}\frac{\partial b}{\partial a}
+
\bar{c}\frac{\partial c}{\partial a}.
$$

So we cannot finish computing \(\bar{a}\) until we have already propagated backward through both \(b\) and \(c\).

A reverse topological ordering guarantees exactly that:

Forward topological order:
x, a, b, c, y

Reverse topological order:
y, c, b, a, x


The exact relative ordering of b and c does not matter, but both must come before a during the reverse pass.

### What topological order guarantees

For every graph edge

$$
u \longrightarrow v
$$


the forward pass processes u before v, because v needs the value of u.

The reverse pass processes v before u, because u's adjoint receives a contribution from v:

$$
\bar{u} \mathrel{+}= \bar{v}\frac{\partial v}{\partial u}.
$$

Thus reverse mode uses the recurrence

$$
\bar{u}
=
\sum_{v \in \operatorname{children}(u)}
\bar{v}\frac{\partial v}{\partial u}.
$$

and reverse topological order ensures every 
v
ˉ
 is available before its contribution is added to \bar{u}.


### Why ordinary program execution often gives the topsort automatically

For a straight-line computation such as

```Python
a = x * x
b = sin(a)
c = a + b
y = c * c
```

the execution order is already a topological ordering:

a, b, c, y

An AD system can record these operations on a tape. Reverse mode then traverses the tape backward:
y, c, b, a

So many reverse-mode implementations do not explicitly run a graph topological-sort algorithm. They get a valid topological ordering from the order in which operations were executed.

### The key idea

It is not that differentiation intrinsically requires a topsort. It is that reverse accumulation has dependency constraints:

    Before propagating through a node, collect all derivative contributions arriving from nodes that depend on it.

Reverse topological traversal is the systematic way to satisfy those constraints.




---


okay using topsort is just extra safety when doing "normal" vector calculus I suppose?

Does it have anything to do with efficiency of the AD in terms of algorithmically saving from evaluating some terms more than once?

Not quite. Topological ordering is not just “extra safety.” It expresses the actual dependency order required by reverse-mode accumulation.

Suppose a value a feeds two later calculations:

```
      ┌→ b ─┐
a ────┤     ├→ y
      └→ c ─┘
```

Then

$$
\bar{a}
=
\bar{b}\frac{\partial b}{\partial a}
+
\bar{c}\frac{\partial c}{\partial a}.
$$

and then propagates once from a to x:

$$
\bar{x}
=
\bar{a}\frac{\partial a}{\partial x}.
$$

So yes: shared upstream calculations are processed once after their downstream derivative contributions have been summed.

This is essentially dynamic programming on the computational graph.

What the topsort itself contributes

The topological ordering ensures that:

1. every node's downstream adjoints are available before that node is processed;
2. each node can normally be processed once;
3. all path contributions are accumulated without explicitly enumerating paths;
4. shared subexpressions do not cause repeated recursive differentiation.

But topsort does not magically eliminate repeated computation on its own. The savings come from the combination of:

- representing the computation as a DAG;
- storing intermediate forward values;
- storing one accumulated adjoint per node;
- traversing nodes in reverse topological order.

For a graph with V nodes and E dependency edges, the reverse pass is typically approximately

$$O(V + E)$$

rather than being proportional to the potentially enormous number of distinct paths through the graph.

One subtle point: in tape-based AD, the normal forward execution order is already a topological order. The system often does not explicitly run a topsort algorithm—it simply records operations and walks the tape backward.