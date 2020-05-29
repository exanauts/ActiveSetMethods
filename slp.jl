#Importing Packages
using JuMP, Ipopt, PowerModels
println("Packages were successfully imported")


#Defining Parameters
eps=0.0001; itr_max=2



#SLP function
function slp(network_data)
	#Initialization
	# TODO: Add some initialization
	itr=0; err=1;
	while(err>eps && itr<itr_max) itr+=1;
		lp=Model(with_optimizer(Ipopt.Optimizer, OutputFlag=0));
		@variable(lp, x[1:2]>=0)
		@variable(lp, y[1:2]>=0)

		@constraint(lp, [i=1:2], x[i]+y[j] >= i*10);
		
		@objective(lp, Min, cx*sum(x)+cy*sum(y));
		@time JuMP.optimize!(lp);
	  	println("Objective Value: ", JuMP.objective_value(lp), "(",JuMP.termination_status(lp),")");
	end
end



#Read Network Data
network_data = PowerModels.parse_file("cases/case3.m")
println("Network Data Read Successfully")



