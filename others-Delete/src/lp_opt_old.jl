#=
The model solves the following minimization problem.
    
      min   cx
      s.b.  A1x + b1 = 0  
            A2x + b2 <= 0

x  -> n x 1
c -> 1 x n
A1 -> m1 x n
b1 -> m1 x 1
A2 -> m2 x n
b1 -> m2 x 1

n=4;
m1=3;
m2=2;

c = rand(n);
A1 = rand(m1,n);
b1 = rand(m1);
A2 = rand(m2,n);
b2 = rand(m2);
=#


function solve_lp(model, n, m1,m2, c, A1,b1,A2,b2,min_max=0)
#model = Gurobi.Optimizer();
sense = MOI.MIN_SENSE
if min_max != 0
	sense = MOI.MAX_SENSE
end



c = convert(Array{Float64}, c);
A1 = convert(Array{Float64}, A1);
b1 = convert(Array{Float64}, b1);
A2 = convert(Array{Float64}, A2);
b2 = convert(Array{Float64}, b2);

println("c: ", c);
println("A1: ", A1);
println("b1: ", b1);
println("A2: ", A2);
println("b2: ", b2);

x = MOI.add_variables(model, n)
MOI.set(model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0.0))
MOI.set(model, MOI.ObjectiveSense(), sense)


for i in 1:m1
        MOI.Utilities.normalize_and_add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A1[i,:],x), b1[i]),
                                 MOI.EqualTo(0.0))
end

for i in 1:m2
        MOI.Utilities.normalize_and_add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A2[i,:],x), b2[i]),
                                 MOI.LessThan(0.0))
end



MOI.optimize!(model);

status = MOI.get(model, MOI.TerminationStatus());

Xsol = zeros(n);
s = 1;


if status == MOI.OPTIMAL
	Xsol = MOI.get(model, MOI.VariablePrimal(), x);
	println(Xsol);
end

return(Xsol, s);
end









