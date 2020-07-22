push!(LOAD_PATH, "../src");
using JuMP
using Ipopt

solver = Ipopt.Optimizer

model = Model(solver);

@variable(model, X);
@objective(model, Min, X^2 + X);
@NLconstraint(model, X^2 - X == 2);

println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

println("Xsol = ",JuMP.value.(X));
