push!(LOAD_PATH, ".");
using JuMP
using Slopt

n=3
m=4

model = Model(with_optimizer(Slopt.Optimizer));


@variable(model, X[1:n, 1:n]);
@variable(model, Y[1:m]);
@variable(model, Z[1:n+m]<=10);

@objective(model, Max, 0.9 * sum(X) + 0.16 * sum(Y) + 0.49 * sum(Z) + X[1]*Z[1]);

@constraint(model, 25 * sum(X) + 35 * sum(Y) + 45 * sum(Z)<= 10000);
@constraint(model, [i=1:m], Y[i]*Z[i] == 4 * i);
println("________________________________________");
print(model);
println("________________________________________");
JuMP.optimize!(model);

