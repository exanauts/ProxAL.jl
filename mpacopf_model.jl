using JuMP
using Ipopt

function init_x(m::JuMP.Model, opfdata::OPFData)
    T = size(opfdata.Pd, 2)
    bus = opfdata.buses
    gen = opfdata.generators
    num_buses = length(bus)
    num_gens = length(gen)

    Vm = zeros(T, num_buses)
    for b=1:num_buses
        Vm[:,b] .= 0.5*(bus[b].Vmax + bus[b].Vmin)
    end
    Va = bus[opfdata.bus_ref].Va * ones(T, num_buses)
    setvalue(m[:Vm], Vm)
    setvalue(m[:Va], Va)

    Pg = zeros(T, num_gens)
    Qg = zeros(T, num_gens)
    for g=1:num_gens
        Pg[:,g] .= 0.5*(gen[g].Pmax + gen[g].Pmin)
        Qg[:,g] .= 0.5*(gen[g].Qmax + gen[g].Qmin)
    end
    setvalue(m[:Pg], Pg)
    setvalue(m[:Qg], Qg)

    Sl = zeros(T, num_gens)
    for g=1:num_gens
        Sl[2:T,g] .= gen[g].ramp_agc
    end
    setvalue(m[:Sl], Sl)
end

function get_mpmodel(opfdata::OPFData; has_ramping::Bool = true)

    m = Model()

    # Shortcuts
    baseMVA = opfdata.baseMVA
    busref = opfdata.bus_ref
    bus = opfdata.buses
    line = opfdata.lines
    gen = opfdata.generators
    busdict = opfdata.BusIdx
    frombus = opfdata.FromLines
    tobus = opfdata.ToLines
    bus2gen = opfdata.BusGenerators
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(line, bus, baseMVA)

    Pd = opfdata.Pd
    Qd = opfdata.Qd
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
        sum(  gen[g].coeff[gen[g].n-2]*(baseMVA*Pg[t,g])^2
            + gen[g].coeff[gen[g].n-1]*(baseMVA*Pg[t,g])
            + gen[g].coeff[gen[g].n  ] for t=1:T,g=1:num_gens))

    # Ramping up/down constraints
    if has_ramping
        Pg_ramp = m[:Pg]

        @constraint(m, ramping[t=2:T,g=1:num_gens],
                        Pg_ramp[t-1,g] - Pg_ramp[t,g] + Sl[t,g] == gen[g].ramp_agc)
    end

    # Power flow constraints: real part

    @NLconstraint(m, pfreal[t=1:T,b=1:num_buses],
        (sum(YffR[l] for l in frombus[b])
            + sum(YttR[l] for l in tobus[b])
                + YshR[b])*Vm[t,b]^2
        + sum(Vm[t,b]*Vm[t,busdict[line[l].to]]*
            (YftR[l]*cos(Va[t,b]-Va[t,busdict[line[l].to]])
                + YftI[l]*sin(Va[t,b]-Va[t,busdict[line[l].to]]))
            for l in frombus[b])
            + sum(Vm[t,b]*Vm[t,busdict[line[l].from]]*
                (YtfR[l]*cos(Va[t,b]-Va[t,busdict[line[l].from]])
                    + YtfI[l]*sin(Va[t,b]-Va[t,busdict[line[l].from]]))
                for l in tobus[b])
                - (sum(baseMVA*Pg[t,g] for g in bus2gen[b]) - (Pd[b,t])) / baseMVA
                == 0)

    # Power flow constraints: imaginary part
    @NLconstraint(m, pfimag[t=1:T,b=1:num_buses],
        (sum(-YffI[l] for l in frombus[b])
            + sum(-YttI[l] for l in tobus[b])
            - YshI[b])*Vm[t,b]^2
        + sum(Vm[t,b]*Vm[t,busdict[line[l].to]]*
            (-YftI[l]*cos(Va[t,b]-Va[t,busdict[line[l].to]])
                + YftR[l]*sin(Va[t,b]-Va[t,busdict[line[l].to]]))
            for l in frombus[b])
            + sum(Vm[t,b]*Vm[t,busdict[line[l].from]]*
                (-YtfI[l]*cos(Va[t,b]-Va[t,busdict[line[l].from]])
                    + YtfR[l]*sin(Va[t,b]-Va[t,busdict[line[l].from]]))
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
        Yff_abs2[i] = YffR[l]^2 + YffI[l]^2
        Yft_abs2[i] = YftR[l]^2 + YftI[l]^2
        Yre[i] = YffR[l]*YftR[l] + YffI[l]*YftI[l]
        Yim[i] = -YffR[l]*YftI[l] + YffI[l]*YftR[l]
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
        Ytf_abs2[i] = YtfR[l]^2 + YtfI[l]^2
        Ytt_abs2[i] = YttR[l]^2 + YttI[l]^2
        Yre[i] = YtfR[l]*YttR[l] + YtfI[l]*YttI[l]
        Yim[i] = -YtfR[l]*YttI[l] + YtfI[l]*YttR[l]
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

function solve_mpmodel(m::JuMP.Model, opfdata::OPFData)
    T = size(opfdata.Pd, 2)
    init_x(m, opfdata)
    setsolver(m, IpoptSolver(print_level=1))
    status = solve(m)

    if status != :Optimal
        println("Stat is not optimal: ", status)
        return
    end

    #
    # Optimal primal solution
    #
    primal = initializePrimalSolution(opfdata, T; sc = false)
    for t=1:T
        for g=1:length(opfdata.generators)
            primal.PG[t,g] = getvalue(m[:Pg][t,g])
            primal.QG[t,g] = getvalue(m[:Qg][t,g])
            primal.SL[t,g] = getvalue(m[:Sl][t,g])
        end
        for b=1:length(opfdata.buses)
            primal.VM[t,b] = getvalue(m[:Vm][t,b])
            primal.VA[t,b] = getvalue(m[:Va][t,b])
        end
    end


    #
    # Optimal dual solution
    #
    dual = initializeDualSolution(opfdata, T; sc = false)
    for t=2:T
        for g=1:length(opfdata.generators)
            dual.λ[t,g] = internalmodel(m).inner.mult_g[linearindex(m[:ramping][t,g])]
        end
    end


    return primal, dual
end

function solve_mpmodel(m::JuMP.Model, opfdata::OPFData, t::Int;
        initial_x::mpPrimalSolution = nothing, initial_λ::mpDualSolution = nothing, params::AlgParams)
    Pg = m[:Pg]
    Qg = m[:Qg]
    Vm = m[:Vm]
    Va = m[:Va]
    Sl = m[:Sl]
    gen = opfdata.generators
    bus = opfdata.buses

    #
    # Initial point
    #
    if initial_x == nothing
        for g=1:length(gen)
            setvalue(Pg[g], 0.5*(gen[g].Pmax + gen[g].Pmin))
            setvalue(Qg[g], 0.5*(gen[g].Qmax + gen[g].Qmin))
        end
        for b=1:length(bus)
            setvalue(Vm[b], 0.5*(bus[b].Vmax + bus[b].Vmin))
            setvalue(Va[b], bus[opfdata.bus_ref].Va)
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

function get_mpmodel(opfdata::OPFData, t::Int; params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)

    m = Model(solver=IpoptSolver(print_level=1))

    # Shortcuts
    baseMVA = opfdata.baseMVA
    busref = opfdata.bus_ref
    bus = opfdata.buses
    line = opfdata.lines
    gen = opfdata.generators
    busdict = opfdata.BusIdx
    frombus = opfdata.FromLines
    tobus = opfdata.ToLines
    bus2gen = opfdata.BusGenerators
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(line, bus, baseMVA)

    Pd = opfdata.Pd
    Qd = opfdata.Qd
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
    penalty = penalty_expression(m, opfdata, t; params = params, dual = dual, primal = primal)

    @objective(m, Min, penalty +
        sum(  gen[g].coeff[gen[g].n-2]*(baseMVA*Pg[g])^2
            + gen[g].coeff[gen[g].n-1]*(baseMVA*Pg[g])
            + gen[g].coeff[gen[g].n  ] for g=1:num_gens))


    # Power flow constraints: real part

    @NLconstraint(m, pfreal[b=1:num_buses],
        (sum(YffR[l] for l in frombus[b])
            + sum(YttR[l] for l in tobus[b])
                + YshR[b])*Vm[b]^2
        + sum(Vm[b]*Vm[busdict[line[l].to]]*
            (YftR[l]*cos(Va[b]-Va[busdict[line[l].to]])
                + YftI[l]*sin(Va[b]-Va[busdict[line[l].to]]))
            for l in frombus[b])
            + sum(Vm[b]*Vm[busdict[line[l].from]]*
                (YtfR[l]*cos(Va[b]-Va[busdict[line[l].from]])
                    + YtfI[l]*sin(Va[b]-Va[busdict[line[l].from]]))
                for l in tobus[b])
                - (sum(baseMVA*Pg[g] for g in bus2gen[b]) - (Pd[b,t])) / baseMVA
                == 0)

    # Power flow constraints: imaginary part
    @NLconstraint(m, pfimag[b=1:num_buses],
        (sum(-YffI[l] for l in frombus[b])
            + sum(-YttI[l] for l in tobus[b])
            - YshI[b])*Vm[b]^2
        + sum(Vm[b]*Vm[busdict[line[l].to]]*
            (-YftI[l]*cos(Va[b]-Va[busdict[line[l].to]])
                + YftR[l]*sin(Va[b]-Va[busdict[line[l].to]]))
            for l in frombus[b])
            + sum(Vm[b]*Vm[busdict[line[l].from]]*
                (-YtfI[l]*cos(Va[b]-Va[busdict[line[l].from]])
                    + YtfR[l]*sin(Va[b]-Va[busdict[line[l].from]]))
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
        Yff_abs2[i] = YffR[l]^2 + YffI[l]^2
        Yft_abs2[i] = YftR[l]^2 + YftI[l]^2
        Yre[i] = YffR[l]*YftR[l] + YffI[l]*YftI[l]
        Yim[i] = -YffR[l]*YftI[l] + YffI[l]*YftR[l]
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
        Ytf_abs2[i] = YtfR[l]^2 + YtfI[l]^2
        Ytt_abs2[i] = YttR[l]^2 + YttI[l]^2
        Yre[i] = YtfR[l]*YttR[l] + YtfI[l]*YttI[l]
        Yim[i] = -YtfR[l]*YttI[l] + YtfI[l]*YttR[l]
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

function penalty_expression(m::JuMP.Model, opfdata::OPFData, t::Int; params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)
    Pg = m[:Pg]
    Qg = m[:Qg]
    Vm = m[:Vm]
    Va = m[:Va]
    Sl = m[:Sl]
    gen = opfdata.generators

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
            if t < size(opfdata.Pd, 2)
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
        for b=1:length(opfdata.buses)
            penalty += 0.5*params.τ*(Vm[b] - primal.VM[t,b])^2
            penalty += 0.5*params.τ*(Va[b] - primal.VA[t,b])^2
        end
    end

    return penalty
end
