using Pkg
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("PowerModels")
println("Packages were successfully installed")
import JuMP, Ipopt, PowerModels
println("Packages were successfully imported")

run_ac_opf("matpower/case3.m", with_optimizer(Ipopt.Optimizer))

