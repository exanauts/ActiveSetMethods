# SuccessiveLinearApproximation

## Using ProxSDP with JuMP

For example, consider the semidefinite programming relaxation of the [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf) problem
```
    max   0.25 * W•X
    s.t.  diag(X) = 1,
          X ≽ 0,
```
This problem can be solved by the following code using **ProxSDP** and [JuMP](https://github.com/JuliaOpt/JuMP.jl).
