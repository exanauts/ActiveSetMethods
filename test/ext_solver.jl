
optimizer_solver = optimizer_with_attributes(
    ActiveSetMethods.Optimizer,
    "external_optimizer" => GLPK.Optimizer,
    "algorithm" => "SLP-LS",
    "OutputFlag" => 0,
)

model = Model(optimizer_solver)

@variable(model, X);
@variable(model, Y);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);
@NLconstraint(model, X * Y == 1);
@NLconstraint(model, X * Y >= 0);
@constraint(model, X >= -2);

JuMP.optimize!(model);

xsol = JuMP.value.(X)
ysol = JuMP.value.(Y)
status = termination_status(model)
