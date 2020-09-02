using ActiveSetMethods, GLPK
using JuMP

#model = Model(optimizer_with_attributes(solver, "lp_solver" => GLPK.Optimizer()));
model = Model(ActiveSetMethods.Optimizer);
set_optimizer_attribute(model, "LP_solver", GLPK.Optimizer)

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
println("Ysol = ",JuMP.value.(Y));

println("Status: ", termination_status(model));
