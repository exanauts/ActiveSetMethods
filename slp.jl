#using Pkg
#Pkg.add("JuMP")
#Pkg.add("Ipopt")
#Pkg.add("PowerModels")
#println("Packages were successfully installed")
using JuMP, Ipopt, PowerModels
println("Packages were successfully imported")

#run_ac_opf("case3.m", with_optimizer(Ipopt.Optimizer))

network_data = PowerModels.parse_file("case3.m")

pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)

print(pm.model)

result = optimize_model!(pm, optimizer=with_optimizer(Ipopt.Optimizer))
