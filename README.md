# Slopt

Slopt is a nonlinear solver based on sequential linear programming method. 

## Using Slopt with JuMP

For example, consider the following quadratic optimization problem
```
        h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x
        max   x^2 + 0.25 * W•X
        s.t.  diag(X) = 1,
        X ≽ 0,
```
This problem can be solved by the following code using **Slopt** and [JuMP](https://github.com/JuliaOpt/JuMP.jl).
```julia
# Load packages
using ProxSDP, JuMP, LinearAlgebra

# Number of vertices
n = 4
# Graph weights
W = [18.0  -5.0  -7.0  -6.0
     -5.0   6.0   0.0  -1.0
     -7.0   0.0   8.0  -1.0
     -6.0  -1.0  -1.0   8.0]

# Build Max-Cut SDP relaxation via JuMP
model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true))
@variable(model, X[1:n, 1:n], PSD)
@objective(model, Max, 0.25 * dot(W, X))
@constraint(model, diag(X) .== 1)

# Solve optimization problem with ProxSDP
JuMP.optimize!(model)

# Retrieve solution
Xsol = JuMP.value.(X)
```

