using ActiveSetMethods, GLPK
using JuMP

model = Model(ActiveSetMethods.Optimizer);
set_optimizer_attribute(model, "external_optimizer", GLPK.Optimizer())

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

xsol = JuMP.value.(X)
ysol = JuMP.value.(Y)
status = termination_status(model)

println("Xsol = ", xsol);
println("Ysol = ", ysol);

println("Status: ", status);
