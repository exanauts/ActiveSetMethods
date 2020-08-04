![ActiveSetMethods](https://github.com/exanauts/ActiveSetMethods/blob/master/src/logo/logo.png "ActiveSetMethods")
---
# ActiveSetMethods

ActiveSetMethods (Active Set Methods) is a nonlinear solver based on various iterative methods and algorihms such as linear search and trust region algorithms using sequential linear programming (SLP) and Quadratic Sequential Programing (QLP) methods. 

## Using ActiveSetMethods with JuMP

For example, consider the following quadratic optimization problem
```
        min   x^2 + x 
        s.t.  x^2 - x = 2
```
This problem can be solved by the following code using [ActiveSetMethods](https://github.com/ssadat/Nopt) and [JuMP](https://github.com/JuliaOpt/JuMP.jl). 
```julia
# Load packages
using ActiveSetMethods, JuMP

# Number of variables
n = 1

# Build nonlinear problem model via JuMP
model = Model(with_optimizer(ActiveSetMethods.Optimizer))
@variable(model, x)
@objective(model, Min, x^2 + x)
@NLconstraint(model, x^2 - x == 2)

# Solve optimization problem with Nlopt
JuMP.optimize!(model)

# Retrieve solution
Xsol = JuMP.value.(X)
```

