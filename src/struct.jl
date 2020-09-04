mutable struct NloptProblem
    #ef::Ptr{Cvoid}  # Reference to the internal data structure
    n::Int  # Num vars
    m::Int  # Num cons
    x::Vector{Float64}  # Starting and final solution
    x_L::Vector{Float64} # Variables Lower Bound
    x_U::Vector{Float64} # Variables Upper Bound
    g::Vector{Float64}  # Final constraint values
    g_L::Vector{Float64} # Constraints Lower Bound
    g_U::Vector{Float64} # Constraints Upper Bound
    j_str::Array{Tuple{Int64,Int64}}
    h_str::Array{Tuple{Int64,Int64}}
    mult_g::Vector{Float64} # lagrange multipliers on constraints
    mult_x_L::Vector{Float64} # lagrange multipliers on lower bounds
    mult_x_U::Vector{Float64} # lagrange multipliers on upper bounds
    obj_val::Float64  # Final objective
    status::Int  # Final status

    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function

    eval_h  # Can be nothing
    intermediate  # Can be nothing

    # For MathProgBase
    sense::Symbol

    parameters::Parameters

    function NloptProblem(n, x_L, x_U, m, g_L, g_U, j_sparsity, h_sparsity,
                          eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h,
                          parameters)
        return new(n, m, zeros(Float64, n), x_L, x_U, zeros(Float64, m), g_L, g_U,
                   j_sparsity, h_sparsity, zeros(Float64,m), zeros(Float64,n),
                   zeros(Float64,n), 0.0, -5, eval_f, eval_g, eval_grad_f,
                   eval_jac_g, eval_h, nothing, :Min,
                   parameters)
    end
end

# From Ipopt/src/Interfaces/IpReturnCodes_inc.h
ApplicationReturnStatus = Dict(
0=>:Solve_Succeeded,
1=>:Solved_To_Acceptable_Level,
2=>:Infeasible_Problem_Detected,
3=>:Search_Direction_Becomes_Too_Small,
4=>:Diverging_Iterates,
5=>:User_Requested_Stop,
6=>:Feasible_Point_Found,
-1=>:Maximum_Iterations_Exceeded,
-2=>:Restoration_Failed,
-3=>:Error_In_Step_Computation,
-4=>:Maximum_CpuTime_Exceeded,
-5=>:Optimize_not_called,
-6=>:Method_not_defined,
-10=>:Not_Enough_Degrees_Of_Freedom,
-11=>:Invalid_Problem_Definition,
-12=>:Invalid_Option,
-13=>:Invalid_Number_Detected,
-100=>:Unrecoverable_Exception,
-102=>:Insufficient_Memory,
-199=>:Internal_Error)
