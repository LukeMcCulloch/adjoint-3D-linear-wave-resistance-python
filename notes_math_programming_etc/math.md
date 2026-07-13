

# derivations

---
### $\frac{d}{dx} log\left(x\right)$

$$
f'\left(x\right) = \lim_{h->0} \frac{ \left[f(x+h) - f(x)\right] }{h}
$$




For log, there's a clean trick: 

$$
\begin{aligned}
log(x+h) - log(x) &= log((x+h)/x) 
\newline
log((x+h)/x) &= log(1 + h/x). 
\end{aligned}
$$

And for small u, log(1+u) ≈ u (this is just the first term of $log(1+u) = u - u²/2 + u³/3 - ... $ as $u→0$, everything past the first term vanishes faster than the first term does). 

So with $u = h/x: log(x+h) - log(x) ≈ h/x$, 

which means $\frac{ \left[f(x+h) - f(x)\right] }{h} = \frac{h/x}{h}$ 


so $\frac{\left[log(x+h)-log(x)\right]}{h} ≈ 1/x $  

the actual code:

```Csharp
def log(x):
    ...
    return Node(float(np.log(x.val)), parents=[(x, 1.0 / x.val)])
```

Here is a worked example:

```Csharp
    x = Node(4.0, [])
    p = log(x)          # p = Node(val=log(4)=1.3863, parents=[(x, 1/4=0.25)])
    backward(p)
    x.grad               # -> 0.25
```



Check: d/dx log(x) at x=4 is 1/4 = 0.25. Matches. x.grad = 0.25 means "if x wiggles a little, p (the thing we called backward() on) wiggles 0.25 as much" — same reading as before, nothing new there.

---
### $ \frac{d}{dx} \arctan \left(x \right)$

Let y = atan(x), so x = tan(y). Differentiate both sides with respect to x, treating y as a function of x (implicit differentiation): 1 = sec²(y) · dy/dx. Using the identity sec²(y) = 1 + tan²(y) = 1 + x² (since x=tan(y)): 1 = (1+x²)·dy/dx, so:

d/dx atan(x) = 1/(1+x²).

Check against the code:

def atan(x):
    ...
    return Node(float(np.arctan(x.val)), parents=[(x, 1.0 / (1.0 + x.val * x.val))])
1.0/(1.0 + x.val*x.val) — matches.

Tiny worked example: q = atan(x) at x=3.0.

x = Node(3.0, [])
q = atan(x)          # q = Node(val=atan(3)=1.249, parents=[(x, 1/(1+9)=0.1)])
backward(q)
x.grad                # -> 0.1
Check: 1/(1+3²) = 1/10 = 0.1. Matches.

Composing with the quotient rule you already have — this is hs_influence's actual first new line
The real code has L = log(num/den), not log of a bare variable. You already know the quotient rule (from panel_geometry's xic=momenteta/area etc.): z=x/y ⟹ dx += dz/y, dy += -z·dz/y. So L=log(num/den) is just two rules chained:

w = num/den (quotient rule: given upstream d_w, push d_num += d_w/den, d_den += -w·d_w/den)
L = log(w) (the rule we just derived: given upstream d_L, push d_w += d_L/w)
Concretely, seed d_L=1.0 (say L is the root): d_w = 1.0/w. Then that d_w feeds into step 1's quotient rule to get d_num and d_den. That's it — no new rule needed, just the two you already have, run back to back. This is exactly what will happen in code once we get there: no separate "derivative of log-of-a-quotient" formula, just log's rule applied to the output of /'s rule, same composition pattern as everything in panel_geometry.

Does this much feel solid, or is there a spot in log's derivation or atan's derivation where you want to stop and poke at it before we go further? Next piece after this would be the angle term's atan((m*e1-h1)/denom1) — same composition idea, just a longer expression inside the atan instead of a bare quotient — but I want to check in here first rather than plow ahead.