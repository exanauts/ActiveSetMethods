#using Pkg
#Pkg.add("JuMP")
#Pkg.add("Ipopt")
#Pkg.add("PowerModels")
#println("Packages were successfully installed")
using JuMP, Ipopt, PowerModels
println("Packages were successfully imported")

#run_ac_opf("case3.m", with_optimizer(Ipopt.Optimizer))

network_data = PowerModels.parse_file("cases/case3.m")

pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)

print(pm.model)

result = optimize_model!(pm, optimizer=with_optimizer(Ipopt.Optimizer))


#=
Number of objective function evaluations             = 14
Number of objective gradient evaluations             = 14
Number of equality constraint evaluations            = 14
Number of inequality constraint evaluations          = 14
Number of equality constraint Jacobian evaluations   = 14
Number of inequality constraint Jacobian evaluations = 14
Number of Lagrangian Hessian evaluations             = 13
Total CPU secs in IPOPT (w/o function evaluations)   =     11.688
Total CPU secs in NLP function evaluations           =      7.580

Total CPU secs in IPOPT (w/o function evaluations)   =      2.311
Total CPU secs in NLP function evaluations           =      1.600
=#
#Another change

