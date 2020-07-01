push!(LOAD_PATH, ".");
using JuMP
using Slopt

model = Model(with_optimizer(Slopt.Optimizer));

@variable(model, X);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);
println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

println("Xsol = ",JuMP.value.(X));


