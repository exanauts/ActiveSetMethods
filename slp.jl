using Pkg
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("PowerModels")
println("Packages were successfully installed")
import JuMP, Ipopt
println("Packages were successfully imported")
