using Plots


# println("length_fx: ", length(fx))
# println("length_px: ", length(px))

# end_iter = 100 #length(fx)
# num_var = 28 #length(px) / length(fx)
# num_const = 32 #Int(length(lamx) / length(fx))

# export start_iter, end_iter, num_var, num_const

#=
for i=start_iter:end_iter
    fx[i] = (fx[i] == 0) ? 1e-8 : fx[i];
    normEx[i] = (normEx[i] == 0) ? 1e-8 : normEx[i];
    Phix[i] = (Phix[i] == 0) ? 1e-8 : Phix[i];
end=#

# rg = start_iter:end_iter
#
# Pxt = reshape(px[start_iter*num_var-num_var+1:end_iter*num_var], num_var,end_iter-start_iter+1)
# alphaPxt = reshape(alphapx[start_iter*num_var-num_var+1:end_iter*num_var], num_var,end_iter-start_iter+1)
# lamxt = reshape(lamx[start_iter*num_const-num_const+1:end_iter*num_const], num_const,end_iter-start_iter+1)

function plot_error_components()
	start_iter = 1
	end_iter = 100 #length(fx)
	num_var = 28 #length(px) / length(fx)
	num_const = 32 #Int(length(lamx) / length(fx))
	rg = start_iter:end_iter
	Pxt = reshape(px[start_iter*num_var-num_var+1:end_iter*num_var], num_var,end_iter-start_iter+1)
	alphaPxt = reshape(alphapx[start_iter*num_var-num_var+1:end_iter*num_var], num_var,end_iter-start_iter+1)
	lamxt = reshape(lamx[start_iter*num_const-num_const+1:end_iter*num_const], num_const,end_iter-start_iter+1)
	p1 = plot(rg, errx[rg], yaxis = ("err"), label = "mu_numerator", lw = 3)
	p2 = plot(rg, normDfx[rg], yaxis = ("norm(df(x))", :log), label = "norm(df(x))", lw = 3)
	p3 = plot(rg, normLamx[rg], yaxis = ("norm(lambda)", :log), label = "norm(lam)", lw = 3)
	p4 = plot(rg, normdCx[rg], yaxis = ("norm(dC)", :log), label = "norm(lam)", lw = 3)
	plot(p1, p2, p3, p4, layout = (2, 2), legend=false)
	return(p1, p2, p3, p4)
end


#=

p1 = plot(rg, mux[rg], yaxis = ("mu", :log), label = "mu", lw = 3)

p2 = plot(rg, alphax[rg], yaxis = ("alpha", :log), label = "alphax", lw = 3)

# p3 = plot(rg, mu_numerator[rg], yaxis = ("mu_numerator"), label = "mu_numerator", lw = 3)
p3 = plot(rg, errx[rg], yaxis = ("err"), label = "mu_numerator", lw = 3)

p4 = plot(rg, mu_RHS[rg], yaxis = ("mu_RHS"), label = "mu_RHS", lw = 3)

p5 = plot(rg, -Dx[rg], yaxis = ("[-D]", :log), label = "[-D]", lw = 3)

p6 = plot(rg, fx[rg], yaxis = ("f(x)", :log), label = "f(x)", lw = 3)

p7 = plot(rg, normEx[rg], yaxis = ("||E||", :log), label = "||E||", lw = 3)

p8 = plot(rg, Phix[rg], yaxis = ("Phi(x)", :log), label = "Phi(x)", lw = 3)

p9 = plot(rg, lamxt', yaxis = ("lam_p"), lw = 3)

p10 = plot(rg, alphaPxt', yaxis = ("alpha * p"), lw = 3)

p11 = plot(rg, normDfx[rg], yaxis = ("norm(df(x))", :log), label = "norm(df(x))", lw = 3)
p12 = plot(rg, normLamx[rg], yaxis = ("norm(lambda)", :log), label = "norm(lam)", lw = 3)

plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, layout = (6, 2), legend=false)

=#

# plot(p9, p10, layout = (2, 1), legend=false)
