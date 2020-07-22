mutable struct SloptProblem
    ref::Ptr{Cvoid}  # Reference to the internal data structure
    n::Int  # Num vars
    m::Int  # Num cons
    x::Vector{Float64}  # Starting and final solution
    g::Vector{Float64}  # Final constraint values
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

    function SloptProblem(ref::Ptr{Cvoid}, n, m,
                          eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
        prob = new(ref, n, m, zeros(Float64, n), zeros(Float64, m), zeros(Float64,m),
                   zeros(Float64,n), zeros(Float64,n), 0.0, 0,
                   eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nothing,
                   :Min)
        # Free the internal IpoptProblem structure when
        # the Julia IpoptProblem instance goes out of scope
        finalizer(freeProblem, prob)
        return prob
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
-10=>:Not_Enough_Degrees_Of_Freedom,
-11=>:Invalid_Problem_Definition,
-12=>:Invalid_Option,
-13=>:Invalid_Number_Detected,
-100=>:Unrecoverable_Exception,
-101=>:NonIpopt_Exception_Thrown,
-102=>:Insufficient_Memory,
-199=>:Internal_Error)
