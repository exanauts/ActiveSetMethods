Base.@kwdef mutable struct Parameters
    mode::String = "Normal"           # If Debug it will allow printing some useful information including collecting values for analysis parameters.
    method::String = "SLP"            # Defines the method -- either SLP or SQP (SQP hasn't been implemented yet)
    algorithm::String = "Line Search" # Defines the algorithm -- either Line Search or Trust Region Method (Trust Region algorithm hasn't been implemented yet)

    # Defines the external solver for suproblems
    external_optimizer::Union{Nothing,MOI.AbstractOptimizer} = nothing
    
    # Whether to use approximation (limited-memory) or Hessian
    hessian_approximation::String = "Hessian"

    # Algorithmic parameters
    tol_residual::Float64 = 1.e-6 # tolerance for Kuhn-Tucker residual
    tol_infeas::Float64 = 1.e-6   # tolerance for constraint violation
    max_iter::Int = 1000          # Defines the maximum number of iterations
    time_limit::Float64 = Inf     # Defines the time limit for the solver. (This hasn't been implemented yet)
    mu::Float64 = 1000.0          # penalty parameter
    max_mu::Float64 = 1.e+6       # maximum mu value allowed
    rho::Float64 = 0.8            # directional derivative parameter defined in (0,1)
    eta::Float64 = 0.4            # descent step test parameter defined in (0,0.5)
    tau::Float64 = 0.9            # line search step decrease parameter defined in (0,1)
    min_alpha::Float64 = 1.e-6    # minimum step size
    tr_size::Float64 = 1000.0     # trust region size
end

function get_parameter(params::Parameters, pname::String)
    return getfield(params, Symbol(pname))
end

function set_parameter(params::Parameters, pname::String, val)
    setfield!(params, Symbol(pname), val)
    return nothing
end
