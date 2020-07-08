push!(LOAD_PATH, ".");
using JuMP
using Slopt 

solver = Slopt.Optimizer

model = Model(solver);

#model.lp_solver = Ipopt.Optimizer()
#lp = Model(Ipopt.Optimizer)
#MOI.set(model, lp)

@variable(model, X);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);

println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

println("Xsol = ",JuMP.value.(X));

