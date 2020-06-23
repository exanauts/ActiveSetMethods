# Slopt

Slopt is a nonlinear solver based on sequential linear programming method. 

## Using Slopt with JuMP

For example, consider the following quadratic optimization problem
```
        min   x2 + x 
        s.t.  x2 - x = 2
```
This problem can be solved by the following code using **Slopt** and [JuMP](https://github.com/JuliaOpt/JuMP.jl).
```julia
# Load packages
using Slopt, JuMP, LinearAlgebra

# Number of variables
n = 1

# Build nonlinear problem model via JuMP
model = Model(with_optimizer(Slopt.Optimizer))
@variable(model, x)
@objective(model, Min, x^2 + x)
@NLconstraint(model, x^2 - x == 2)

# Solve optimization problem with Slopt
JuMP.optimize!(model)

# Retrieve solution
Xsol = JuMP.value.(X)
```

