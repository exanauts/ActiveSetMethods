push!(LOAD_PATH, ".");
using JuMP
using Slopt

n=3
m=4

model = Model(with_optimizer(Slopt.Optimizer));


#@variable(model, X[1:n, 1:n]);
#@variable(model, X[1:n]);
@variable(model, X);
#@variable(model, Y[1:m]);
#@variable(model, Z[1:n+m]<=10);
@objective(model, Min, X^2 + X);
#@objective(model, Min, 49 * X' * X + 16 * sum(X));
#@objective(model, Min, 60 * X[1] * X[2] + 70 * X[2] * X[3] + 80 * X[3] * X[1] + 40 * X[2] * X[2] + 50 * X[3] * X[3] + 16 * sum(X) + 100);
#@NLobjective(model, Min, 49 * (X[1]-1)^2);
#@objective(model, Min, 16 * sum(X));
#@objective(model, Max, 0.9 * sum(X) + 0.16 * sum(Y) + 0.49 * sum(Z));
@NLconstraint(model, X^2 - X == 2);
@NLconstraint(model, X^2 + X >= 0);
#@constraint(model, 25 * sum(X) + 35 >= 10000);
#@constraint(model, 25 * sum(X) + 35 * sum(Y) + 45 * sum(Z)<= 10000);
#@constraint(model, [i=1:m], Y[i]*Z[i] == 4 * i);
println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

println("Xsol = ",JuMP.value.(X));
#println("Ysol = ",JuMP.value.(Y));
#println("Zsol = ",JuMP.value.(Z));
