using Revise
using ActiveSetMethods
using PowerModels, JuMP, GLPK, Ipopt
# using CPLEX

PowerModels.silence()

# choose an internal LP solver
lp_solver = GLPK.Optimizer
# lp_solver = optimizer_with_attributes(
#     CPLEX.Optimizer,
#     "CPX_PARAM_SCRIND" => 0,
#     "CPX_PARAM_THREADS" => 1,
# )

include("acwr.jl")

build_acp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACPPowerModel, PowerModels.build_opf)
build_acr(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACRPowerModel, PowerModels.build_opf)
build_iv(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), IVRPowerModel, PowerModels.build_opf_iv)
build_dcp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), DCPPowerModel, PowerModels.build_opf_iv)

function run_opf(data_file::String, max_iter::Int = 100, algorithm = "Line Search")
    pm = build_acp(data_file)
    # pm2 = build_acp(data_file)
    # JuMP.@objective(pm2.model, Min, 0)
    init_vars(pm)
    # init_vars_from_ipopt(pm, pm2)
    result = optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "external_optimizer" => lp_solver,
        "max_iter" => max_iter,
        "algorithm" => algorithm,
    ))
    return pm, result
end

include("init_opf.jl")
