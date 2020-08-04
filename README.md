![ActiveSetMethods](https://github.com/exanauts/ActiveSetMethods/blob/master/src/logo/Logo.png "ActiveSetMethods")
---
# Nlopt

Nlopt (Nonlinear Optimizer) is a nonlinear solver based on various iterative methods and algorihms such as linear search and trust region algorithms using sequential linear programming (SLP) and Quadratic Sequential Programing (QLP) methods. 

## Using Nlopt with JuMP

For example, consider the following quadratic optimization problem
```
        min   x^2 + x 
        s.t.  x^2 - x = 2
```
This problem can be solved by the following code using [Nlopt](https://github.com/ssadat/Nlopt) and [JuMP](https://github.com/JuliaOpt/JuMP.jl). 
```julia
# Load packages
using Nlopt, JuMP, LinearAlgebra

# Number of variables
n = 1

# Build nonlinear problem model via JuMP
model = Model(with_optimizer(Nlopt.Optimizer))
@variable(model, x)
@objective(model, Min, x^2 + x)
@NLconstraint(model, x^2 - x == 2)

# Solve optimization problem with Nlopt
JuMP.optimize!(model)

# Retrieve solution
Xsol = JuMP.value.(X)
```

