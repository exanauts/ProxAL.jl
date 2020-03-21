using JuMP
using Ipopt

function init_x(m, circuit, demand; opt::Option=Option())
    num_buses = length(circuit.bus)
    num_gens = length(circuit.gen)
    T = size(demand.pd,2)

    Vm = zeros(T, num_buses)
    for b=1:num_buses
        Vm[:,b] .= 0.5*(circuit.bus[b].Vmax + circuit.bus[b].Vmin)
    end
    Va = circuit.bus[circuit.busref].Va * ones(T, num_buses)
    setvalue(m[:Vm], Vm)
    setvalue(m[:Va], Va)

    Pg = zeros(T, num_gens)
    Qg = zeros(T, num_gens)
    for g=1:num_gens
        Pg[:,g] .= 0.5*(circuit.gen[g].Pmax + circuit.gen[g].Pmin)
        Qg[:,g] .= 0.5*(circuit.gen[g].Qmax + circuit.gen[g].Qmin)
    end
    setvalue(m[:Pg], Pg)
    setvalue(m[:Qg], Qg)

    Sl = zeros(T, num_gens)
    for g=1:num_gens
        Sl[2:T,g] .= circuit.gen[g].ramp_agc
    end
    setvalue(m[:Sl], Sl)
end

function get_mpmodel(circuit, demand; opt::Option=Option())

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
    @variable(m, 0 <= Sl[t=1:T,g=1:num_gens] <= 2Float64(t > 1)*gen[g].ramp_agc)

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

        @constraint(m, ramping[t=2:T,g=1:num_gens],
                        Pg_ramp[t-1,g] - Pg_ramp[t,g] + Sl[t,g] == gen[g].ramp_agc)
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

function solve_mpmodel(m, circuit, demand)
    T = size(demand.pd, 2)
    init_x(m, circuit, demand)
    setsolver(m, IpoptSolver(print_level=1))
    status = solve(m)

    if status != :Optimal
        println("Stat is not optimal: ", status)
        return
    end

    #
    # Optimal primal solution
    #
    primal = initializePrimalSolution(circuit, T)
    for t=1:T
        for g=1:length(circuit.gen)
            primal.PG[t,g] = getvalue(m[:Pg][t,g])
            primal.QG[t,g] = getvalue(m[:Qg][t,g])
            primal.SL[t,g] = getvalue(m[:Sl][t,g])
        end
        for b=1:length(circuit.bus)
            primal.VM[t,b] = getvalue(m[:Vm][t,b])
            primal.VA[t,b] = getvalue(m[:Va][t,b])
        end
    end


    #
    # Optimal dual solution
    #
    dual = initializeDualSolution(circuit, T)
    for t=2:T
        for g=1:length(circuit.gen)
            dual.λ[t,g] = internalmodel(m).inner.mult_g[linearindex(m[:ramping][t,g])]
        end
    end


    return primal, dual
end

function solve_mpmodel(m, circuit, t::Int;
        initial_x::mpPrimalSolution = nothing, initial_λ::mpDualSolution = nothing, params::AlgParams)
    Pg = m[:Pg]
    Qg = m[:Qg]
    Vm = m[:Vm]
    Va = m[:Va]
    Sl = m[:Sl]
    gen = circuit.gen

    #
    # Initial point
    #
    if initial_x == nothing
        for g=1:length(gen)
            setvalue(Pg[g], 0.5*(gen[g].Pmax + gen[g].Pmin))
            setvalue(Qg[g], 0.5*(gen[g].Qmax + gen[g].Qmin))
        end
        for b=1:length(length(circuit.bus))
            setvalue(Vm[b], 0.5*(circuit.bus[b].Vmax + circuit.bus[b].Vmin))
            setvalue(Va[b], circuit.bus[circuit.busref].Va)
        end
    else
        setvalue.(Pg, initial_x.PG[t,:])
        setvalue.(Qg, initial_x.QG[t,:])
        setvalue.(Vm, initial_x.VM[t,:])
        setvalue.(Va, initial_x.VA[t,:])
    end
    #
    # Initial point for slack variables
    #
    if t > 1
        for g=1:length(gen)
            tp = (initial_λ == nothing) ? 0 : (initial_λ.λ[t,g]/params.ρ)
            xp = (initial_x == nothing) ? 0 : (initial_x.PG[t-1,g] - initial_x.PG[t,g])
            setvalue(Sl[g], min(max(0, -xp + gen[g].ramp_agc - tp), 2gen[g].ramp_agc))
        end
    end

    #
    # Solve model
    #
    status = solve(m)

    return m, status
end

function get_mpmodel(circuit, demand, t::Int; opt::Option=Option(), params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)

    m = Model(solver=IpoptSolver(print_level=1))

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

    @variable(m, gen[g].Pmin <= Pg[g=1:num_gens] <= gen[g].Pmax)
    @variable(m, gen[g].Qmin <= Qg[g=1:num_gens] <= gen[g].Qmax)
    @variable(m, bus[b].Vmin <= Vm[b=1:num_buses] <= bus[b].Vmax)
    @variable(m, Va[b=1:num_buses])
    @variable(m, 0 <= Sl[g=1:num_gens] <= 2Float64(t > 1)*gen[g].ramp_agc)

    setlowerbound(Va[busref], bus[busref].Va)
    setupperbound(Va[busref], bus[busref].Va)

    #
    # Penalty terms
    #
    penalty = penalty_expression(m, circuit, t, T; params = params, dual = dual, primal = primal)

    @objective(m, Min, penalty +
        sum(gen[g].coeff2*(baseMVA*Pg[g])^2
            + gen[g].coeff1*(baseMVA*Pg[g])
            + gen[g].coeff0 for g=1:num_gens))


    # Power flow constraints: real part

    @NLconstraint(m, pfreal[b=1:num_buses],
        (sum(yline[l].YffR for l in frombus[b])
            + sum(yline[l].YttR for l in tobus[b])
                + ybus[b].YshR)*Vm[b]^2
        + sum(Vm[b]*Vm[busdict[line[l].to]]*
            (yline[l].YftR*cos(Va[b]-Va[busdict[line[l].to]])
                + yline[l].YftI*sin(Va[b]-Va[busdict[line[l].to]]))
            for l in frombus[b])
            + sum(Vm[b]*Vm[busdict[line[l].from]]*
                (yline[l].YtfR*cos(Va[b]-Va[busdict[line[l].from]])
                    + yline[l].YtfI*sin(Va[b]-Va[busdict[line[l].from]]))
                for l in tobus[b])
                - (sum(baseMVA*Pg[g] for g in bus2gen[b]) - (Pd[b,t])) / baseMVA
                == 0)

    # Power flow constraints: imaginary part
    @NLconstraint(m, pfimag[b=1:num_buses],
        (sum(-yline[l].YffI for l in frombus[b])
            + sum(-yline[l].YttI for l in tobus[b])
            - ybus[b].YshI)*Vm[b]^2
        + sum(Vm[b]*Vm[busdict[line[l].to]]*
            (-yline[l].YftI*cos(Va[b]-Va[busdict[line[l].to]])
                + yline[l].YftR*sin(Va[b]-Va[busdict[line[l].to]]))
            for l in frombus[b])
            + sum(Vm[b]*Vm[busdict[line[l].from]]*
                (-yline[l].YtfI*cos(Va[b]-Va[busdict[line[l].from]])
                    + yline[l].YtfR*sin(Va[b]-Va[busdict[line[l].from]]))
                for l in tobus[b])
                - (sum(baseMVA*Qg[g] for g in bus2gen[b]) - Qd[b,t]) / baseMVA
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

    @NLconstraint(m, flowmaxfrom[i=1:num_linelimits],
        Vm[busdict[line[limind[i]].from]]^2 *
        (Yff_abs2[i]*Vm[busdict[line[limind[i]].from]]^2
            + Yft_abs2[i]*Vm[busdict[line[limind[i]].to]]^2
            + 2*Vm[busdict[line[limind[i]].from]]*Vm[busdict[line[limind[i]].to]]*
            (Yre[i]*cos(Va[busdict[line[limind[i]].from]] - Va[busdict[line[limind[i]].to]])
                - Yim[i]*sin(Va[busdict[line[limind[i]].from]] - Va[busdict[line[limind[i]].to]]))
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

    @NLconstraint(m, flowmaxto[i=1:num_linelimits],
        Vm[busdict[line[limind[i]].to]]^2 *
        (Ytf_abs2[i]*Vm[busdict[line[limind[i]].from]]^2
            + Ytt_abs2[i]*Vm[busdict[line[limind[i]].to]]^2
            + 2*Vm[busdict[line[limind[i]].from]]*Vm[busdict[line[limind[i]].to]]*
            (Yre[i]*cos(Va[busdict[line[limind[i]].from]] - Va[busdict[line[limind[i]].to]])
                -Yim[i]*sin(Va[busdict[line[limind[i]].from]] - Va[busdict[line[limind[i]].to]]))
            ) - flowmax[i] <=0)

    return m
end

function penalty_expression(m, circuit, t::Int, T; params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)
    Pg = m[:Pg]
    Qg = m[:Qg]
    Vm = m[:Vm]
    Va = m[:Va]
    Sl = m[:Sl]
    gen = circuit.gen

    #
    # Penalty terms for ALADIN
    #
    @expression(m, penalty, 0)
    if (params.aladin)
        error("to do: ALADIN for mpacopf")
    #
    # Penalty terms for prox-ALM
    #
    else
        for g=1:length(gen)
            if t > 1
                penalty += dual.λ[t,g]*(+primal.PG[t-1,g] - Pg[g] + Sl[g] - gen[g].ramp_agc)
                penalty += 0.5params.ρ*(+primal.PG[t-1,g] - Pg[g] + Sl[g] - gen[g].ramp_agc)^2
            end
            if t < T
                penalty += dual.λ[t+1,g]*(+Pg[g] - primal.PG[t+1,g] + primal.SL[t+1,g] - gen[g].ramp_agc)
                penalty +=   0.5params.ρ*(+Pg[g] - primal.PG[t+1,g] + primal.SL[t+1,g] - gen[g].ramp_agc)^2
            end
        end
    end

    #
    # the proximal part
    #
    if !iszero(params.τ)
        for g=1:length(gen)
            penalty += 0.5*params.τ*(Pg[g] - primal.PG[t,g])^2
            penalty += 0.5*params.τ*(Qg[g] - primal.QG[t,g])^2
        end
        for b=1:length(circuit.bus)
            penalty += 0.5*params.τ*(Vm[b] - primal.VM[t,b])^2
            penalty += 0.5*params.τ*(Va[b] - primal.VA[t,b])^2
        end
    end

    return penalty
end
