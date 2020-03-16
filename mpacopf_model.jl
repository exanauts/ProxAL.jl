using JuMP
using Ipopt

function get_mpmodel(circuit, demand;
                     opt::Option=Option())

    m = Model()

    # Shortcuts
    baseMVA = circuit.baseMVA
    busref = circuit.busref
    bus = circuit.bus
    line = circuit.line
    gen = circuit.gen
    yline = circuit.yline
    ybus = circuit.ybus
    busdict = circuit.busdict
    frombus = circuit.frombus
    tobus = circuit.tobus
    bus2gen = circuit.bus2gen

    Pd = demand.pd
    Qd = demand.qd
    T = size(Pd,2)

    num_buses = length(bus)
    num_gens = length(gen)
    num_lines = length(line)

    @variable(m, gen[g].Pmin <= Pg[t=1:T,g=1:num_gens] <= gen[g].Pmax)
    @variable(m, gen[g].Qmin <= Qg[t=1:T,g=1:num_gens] <= gen[g].Qmax)
    @variable(m, bus[b].Vmin <= Vm[t=1:T,b=1:num_buses] <= bus[b].Vmax)
    @variable(m, Va[t=1:T,b=1:num_buses])

    for t in 1:T
        setlowerbound(Va[t,busref], bus[busref].Va)
        setupperbound(Va[t,busref], bus[busref].Va)
    end

    @NLobjective(m, Min,
                 sum(gen[g].coeff2*(baseMVA*Pg[t,g])^2
                     + gen[g].coeff1*(baseMVA*Pg[t,g])
                     + gen[g].coeff0 for t=1:T,g=1:num_gens))

    # Ramping up/down constraints
    if opt.has_ramping
        Pg_ramp = m[:Pg]

        @constraint(m, ramping[t=1:T-1,g=1:num_gens],
                    -gen[g].ramp_agc <= Pg_ramp[t+1,g] - Pg_ramp[t,g] <= gen[g].ramp_agc)
    end

    # Power flow constraints: real part

    @NLconstraint(m, pfreal[t=1:T,b=1:num_buses],
                  (sum(yline[l].YffR for l in frombus[b])
                   + sum(yline[l].YttR for l in tobus[b])
                   + ybus[b].YshR)*Vm[t,b]^2
                  + sum(Vm[t,b]*Vm[t,busdict[line[l].to]]*
                        (yline[l].YftR*cos(Va[t,b]-Va[t,busdict[line[l].to]])
                         + yline[l].YftI*sin(Va[t,b]-Va[t,busdict[line[l].to]]))
                        for l in frombus[b])
                  + sum(Vm[t,b]*Vm[t,busdict[line[l].from]]*
                        (yline[l].YtfR*cos(Va[t,b]-Va[t,busdict[line[l].from]])
                         + yline[l].YtfI*sin(Va[t,b]-Va[t,busdict[line[l].from]]))
                        for l in tobus[b])
                  - (sum(baseMVA*Pg[t,g] for g in bus2gen[b]) - (Pd[b,t])) / baseMVA
                  == 0)

    # Power flow constraints: imaginary part
    @NLconstraint(m, pfimag[t=1:T,b=1:num_buses],
                  (sum(-yline[l].YffI for l in frombus[b])
                   + sum(-yline[l].YttI for l in tobus[b])
                   - ybus[b].YshI)*Vm[t,b]^2
                  + sum(Vm[t,b]*Vm[t,busdict[line[l].to]]*
                        (-yline[l].YftI*cos(Va[t,b]-Va[t,busdict[line[l].to]])
                         + yline[l].YftR*sin(Va[t,b]-Va[t,busdict[line[l].to]]))
                        for l in frombus[b])
                  + sum(Vm[t,b]*Vm[t,busdict[line[l].from]]*
                        (-yline[l].YtfI*cos(Va[t,b]-Va[t,busdict[line[l].from]])
                         + yline[l].YtfR*sin(Va[t,b]-Va[t,busdict[line[l].from]]))
                        for l in tobus[b])
                  - (sum(baseMVA*Qg[t,g] for g in bus2gen[b]) - Qd[b,t]) / baseMVA
                  == 0)

    # Line limits
    rateA = getfield.(line, :rateA)
    limind = findall((rateA .!= 0) .& (rateA .< 1.0e10))
    num_linelimits = length(limind)

    Yff_abs2 = zeros(num_linelimits)
    Yft_abs2 = zeros(num_linelimits)
    Yre = zeros(num_linelimits)
    Yim = zeros(num_linelimits)
    flowmax = zeros(num_linelimits)

    for i in 1:num_linelimits
        # Apparent power limits (from bus)
        l = limind[i]
        flowmax[i] = (line[l].rateA / baseMVA)^2
        Yff_abs2[i] = yline[l].YffR^2 + yline[l].YffI^2
        Yft_abs2[i] = yline[l].YftR^2 + yline[l].YftI^2
        Yre[i] = yline[l].YffR*yline[l].YftR + yline[l].YffI*yline[l].YftI
        Yim[i] = -yline[l].YffR*yline[l].YftI + yline[l].YffI*yline[l].YftR
    end

    @NLconstraint(m, flowmaxfrom[t=1:T,i=1:num_linelimits],
                  Vm[t,busdict[line[limind[i]].from]]^2 *
                  (Yff_abs2[i]*Vm[t,busdict[line[limind[i]].from]]^2
                   + Yft_abs2[i]*Vm[t,busdict[line[limind[i]].to]]^2
                   + 2*Vm[t,busdict[line[limind[i]].from]]*Vm[t,busdict[line[limind[i]].to]]*
                   (Yre[i]*cos(Va[t,busdict[line[limind[i]].from]] - Va[t,busdict[line[limind[i]].to]])
                    - Yim[i]*sin(Va[t,busdict[line[limind[i]].from]] - Va[t,busdict[line[limind[i]].to]]))
                   ) - flowmax[i] <= 0)

    Ytf_abs2 = zeros(num_linelimits)
    Ytt_abs2 = zeros(num_linelimits)

    for i in 1:num_linelimits
        # Apparent power limits (to bus)
        l = limind[i]
        Ytf_abs2[i] = yline[l].YtfR^2 + yline[l].YtfI^2
        Ytt_abs2[i] = yline[l].YttR^2 + yline[l].YttI^2
        Yre[i] = yline[l].YtfR*yline[l].YttR + yline[l].YtfI*yline[l].YttI
        Yim[i] = -yline[l].YtfR*yline[l].YttI + yline[l].YtfI*yline[l].YttR
    end

    @NLconstraint(m, flowmaxto[t=1:T,i=1:num_linelimits],
                  Vm[t,busdict[line[limind[i]].to]]^2 *
                  (Ytf_abs2[i]*Vm[t,busdict[line[limind[i]].from]]^2
                   + Ytt_abs2[i]*Vm[t,busdict[line[limind[i]].to]]^2
                   + 2*Vm[t,busdict[line[limind[i]].from]]*Vm[t,busdict[line[limind[i]].to]]*
                   (Yre[i]*cos(Va[t,busdict[line[limind[i]].from]] - Va[t,busdict[line[limind[i]].to]])
                    -Yim[i]*sin(Va[t,busdict[line[limind[i]].from]] - Va[t,busdict[line[limind[i]].to]]))
                   ) - flowmax[i] <=0)

    return m
end

function get_objectivevalue(m, circuit, T)
    baseMVA = circuit.baseMVA
    gen = circuit.gen

    num_gens = length(gen)

    Pg = getvalue(m[:Pg])
    objval = sum(gen[g].coeff2*(baseMVA*Pg[t,g])^2 +
                 gen[g].coeff1*(baseMVA*Pg[t,g]) +
                 gen[g].coeff0 for t=1:T,g=1:num_gens)

    return objval
end

function get_constrviolation(m, d)
    viol = 0

    # Check constraints.
    if MathProgBase.numconstr(m) > 0
	g = zeros(MathProgBase.numconstr(m))
	MathProgBase.eval_g(d, g, m.colVal)

	g_lb, g_ub = JuMP.constraintbounds(m)
	for i=1:length(g_lb)
	    if g_lb[i] != -Inf
		err = max(0, -(g[i] - g_lb[i]))
		if viol < err
		    viol = err
		end
	    end

	    if g_ub[i] != Inf
		err = max(0, g[i] - g_ub[i])
		if viol < err
		    viol = err
		end
	    end
	end
    end

    # Check bound constraints.
    for i=1:MathProgBase.numvar(m)
	if m.colLower[i] != -Inf
	    err = max(0, -(m.colVal[i] - m.colLower[i]))
	    if viol < err
		viol = err
	    end
	end

	if m.colUpper[i] != Inf
	    err = max(0, m.colVal[i] - m.colUpper[i])
	    if viol < err
		viol = err
	    end
	end
    end

    return viol
end
