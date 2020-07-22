push!(LOAD_PATH, "../src");
using JuMP
using Slopt, Gurobi

solver = Slopt.Optimizer

#model = Model(optimizer_with_attributes(solver, "lp_solver" => Gurobi.Optimizer()));
model = Model(solver);
#set_optimizer_attribute(model, "lp_solver", Gurobi.Optimizer)
set_optimizer_attribute(model, "eta", 0.01)

#optimizer_with_attributes(lp_solver=Gurobi.Optimizer())
#set_optimizer_attribute(model, "lp_solver", Gurobi.Optimizer)

#model.lp_solver = Ipopt.Optimizer()
#lp = Model(Ipopt.Optimizer)
#MOI.set(model, lp)

@variable(model, X);
@variable(model, Y);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);
@NLconstraint(model, X*Y == 1);
@NLconstraint(model, X*Y >= 0);
@constraint(model, X >= -2);

println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

println("Xsol = ",JuMP.value.(X));
