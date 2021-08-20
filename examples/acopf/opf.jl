using Revise
using ActiveSetMethods
using PowerModels, JuMP, GLPK, Ipopt

include("acwr.jl")

build_acp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACPPowerModel, PowerModels.build_opf)
build_acr(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), ACRPowerModel, PowerModels.build_opf)
build_iv(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), IVRPowerModel, PowerModels.build_opf_iv)
build_dcp(data_file::String) = instantiate_model(PowerModels.parse_file(data_file), DCPPowerModel, PowerModels.build_opf_iv)

run_opf(data_file::String, max_iter::Int = 100) = run_opf(build_acp(data_file), max_iter)
run_opf(pm::AbstractPowerModel, max_iter::Int = 100) = optimize_model!(pm, optimizer = optimizer_with_attributes(
    ActiveSetMethods.Optimizer, 
    "external_optimizer" => GLPK.Optimizer,
    "max_iter" => max_iter,
    "algorithm" => "Line Search",
    # "algorithm" => "Trust Region",
))

function run_opf_ls(data_file::String, max_iter::Int = 100)
    pm = build_acp(data_file)
    pm2 = build_acp(data_file)
    JuMP.@objective(pm2.model, Min, 0)
    # init_vars(pm)
    init_vars_from_ipopt(pm, pm2)
    optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "external_optimizer" => GLPK.Optimizer,
        "max_iter" => max_iter,
        "algorithm" => "Line Search",
    ))
end

function run_opf_tr(data_file::String, max_iter::Int = 100)
    pm = build_acp(data_file)
    pm2 = build_acp(data_file)
    JuMP.@objective(pm2.model, Min, 0)
    # init_vars(pm)
    init_vars_from_ipopt(pm, pm2)
    optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "external_optimizer" => GLPK.Optimizer,
        "max_iter" => max_iter,
        "algorithm" => "Trust Region",
    ))
end

include("init_opf.jl")

#=
One can run the following:

pm = build_acp("case3.m")
init_vars(pm)
# or 
init_vars_from_ipopt(pm, build_acp("case3.m"))
run_opf(pm)

pm = build_acr("case3.m")

pm = build_iv("case3.m")

=#
