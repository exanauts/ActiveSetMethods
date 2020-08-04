push!(LOAD_PATH, "src");
using JuMP
using ActiveSetMethods, Gurobi, PowerModels

solver = ActiveSetMethods.Optimizer
Options_["LP_solver"]=Gurobi.Optimizer

#run_ac_opf("case3.m", with_optimizer(Ipopt.Optimizer))

network_data = PowerModels.parse_file("cases/case3.m")

pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)



result = optimize_model!(pm, optimizer=solver)
