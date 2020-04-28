using JuMP
using Ipopt

function init_x(opfmodel::JuMP.Model, opfdata::OPFData, T::Int; sc::Bool)
    buses = opfdata.buses
    generators = opfdata.generators
    nbus = length(buses)
    ngen = length(generators)

    Vm = zeros(T, nbus)
    for b=1:nbus
        Vm[:,b] .= 0.5*(buses[b].Vmax + buses[b].Vmin)
    end
    Va = buses[opfdata.bus_ref].Va * ones(T, nbus)
    setvalue(opfmodel[:Vm], Vm)
    setvalue(opfmodel[:Va], Va)

    Pg = zeros(T, ngen)
    Qg = zeros(T, ngen)
    for g=1:ngen
        Pg[:,g] .= 0.5*(generators[g].Pmax + generators[g].Pmin)
        Qg[:,g] .= 0.5*(generators[g].Qmax + generators[g].Qmin)
    end
    setvalue(opfmodel[:Pg], Pg)
    setvalue(opfmodel[:Qg], Qg)

    Sl = zeros(T, ngen)
    for g=1:ngen
        Sl[2:T,g] .= (sc ? generators[g].scen_agc : generators[g].ramp_agc)
    end
    setvalue(opfmodel[:Sl], Sl)
end

function scopf_model(opfdata::OPFData, rawdata::RawData; sc::Bool)

    # number of variable blocks
    T = size(opfdata.Pd, 2)
    lines_off = [Line() for t in 1:T]

    # security-constrained OPF: explicitly add the "base case"
    if sc
        T = length(rawdata.ctgs_arr) + 1
        lines_off = [opfdata.lines[l] for l in rawdata.ctgs_arr]
        lines_off = [Line(); lines_off]
    end


    # the model
    opfmodel = Model(solver=IpoptSolver(print_level=1))


    #shortcuts for compactness
    buses = opfdata.buses
    baseMVA = opfdata.baseMVA
    generators = opfdata.generators
    ngen  = length(generators)
    nbus  = length(buses)

    @variable(opfmodel, generators[i].Pmin <= Pg[1:T,i=1:ngen] <= generators[i].Pmax)
    @variable(opfmodel, generators[i].Qmin <= Qg[1:T,i=1:ngen] <= generators[i].Qmax)
    @variable(opfmodel, buses[i].Vmin <= Vm[1:T,i=1:nbus] <= buses[i].Vmax)
    @variable(opfmodel, Va[1:T,1:nbus])

    # slack variable for ramping constraints
    @variable(opfmodel, 0 <= Sl[t=1:T,g=1:ngen] <= 2Float64(t > 1)*(sc ? generators[g].scen_agc : generators[g].ramp_agc))

    # reference bus voltage angle
    for t in 1:T
        setlowerbound(Va[t,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
        setupperbound(Va[t,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
    end


    # security-constrained opf
    if sc
        # Coupling constraints: power generation@(contingency - base case) <= bounded by ramp factor
        @constraint(opfmodel, ramping[t=2:T,g=1:ngen], Pg[1,g] - Pg[t,g] + Sl[t,g] == generators[g].scen_agc)

        # Minimize only base case cost
        @NLobjective(opfmodel, Min,
            sum(  generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[1,g])^2
                + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[1,g])
                + generators[g].coeff[generators[g].n  ] for g=1:ngen)
        )

    # multi-period opf
    else
        # Ramping up/down constraints
        @constraint(opfmodel, ramping[t=2:T,g=1:ngen], Pg[t-1,g] - Pg[t,g] + Sl[t,g] == generators[g].ramp_agc)

        # Minimize cost over the entire horizon
        @NLobjective(opfmodel, Min,
            sum(  generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[t,g])^2
                + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[t,g])
                + generators[g].coeff[generators[g].n  ] for t=1:T,g=1:ngen)
        )
    end
    



    # constraints for each block
    for t in 1:T
        # Network data
        opfdata_c = opfdata

        # Perturbed network data for contingencies
        if sc
            opfdata_c = opf_loaddata(rawdata; lineOff = lines_off[t])
            Pd = [buses[b].Pd for b in 1:nbus]
            Qd = [buses[b].Qd for b in 1:nbus]
        else
            Pd = opfdata.Pd[:,t]
            Qd = opfdata.Qd[:,t]
        end



        # Network data short-hands
        lines = opfdata_c.lines
        busIdx = opfdata_c.BusIdx
        FromLines = opfdata_c.FromLines
        ToLines = opfdata_c.ToLines
        BusGeners = opfdata_c.BusGenerators
        nline = length(lines)
        YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
            
    
        # Power Balance Equations
        for b in 1:nbus
            #real part
            @NLconstraint(opfmodel, 
                ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[t,b]^2 
                + sum( Vm[t,b]*Vm[t,busIdx[lines[l].to]]  *( YftR[l]*cos(Va[t,b]-Va[t,busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[t,b]-Va[t,busIdx[lines[l].to]]  )) for l in FromLines[b] )  
                + sum( Vm[t,b]*Vm[t,busIdx[lines[l].from]]*( YtfR[l]*cos(Va[t,b]-Va[t,busIdx[lines[l].from]]) + YtfI[l]*sin(Va[t,b]-Va[t,busIdx[lines[l].from]])) for l in ToLines[b]   ) 
                - ( sum(baseMVA*Pg[t,g] for g in BusGeners[b]) - Pd[b]) / baseMVA      # Sbus part
                ==0
            )

            #imaginary part
            @NLconstraint(opfmodel,
                ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[t,b]^2 
                + sum( Vm[t,b]*Vm[t,busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[t,b]-Va[t,busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[t,b]-Va[t,busIdx[lines[l].to]]  )) for l in FromLines[b] )
                + sum( Vm[t,b]*Vm[t,busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[t,b]-Va[t,busIdx[lines[l].from]]) + YtfR[l]*sin(Va[t,b]-Va[t,busIdx[lines[l].from]])) for l in ToLines[b]   )
                - ( sum(baseMVA*Qg[t,g] for g in BusGeners[b]) - Qd[b]) / baseMVA      #Sbus part
                ==0
            )
        end


        # Line flow limits
        nlinelim=0
        for l in 1:nline
            if lines[l].rateA!=0 && lines[l].rateA<1.0e10
                nlinelim += 1
                flowmax=(lines[l].rateA/baseMVA)^2

                #branch apparent power limits (from bus)
                Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
                Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
                @NLconstraint(opfmodel,
                    Vm[t,busIdx[lines[l].from]]^2 *
                    ( Yff_abs2*Vm[t,busIdx[lines[l].from]]^2 + Yft_abs2*Vm[t,busIdx[lines[l].to]]^2 
                    + 2*Vm[t,busIdx[lines[l].from]]*Vm[t,busIdx[lines[l].to]]*(Yre*cos(Va[t,busIdx[lines[l].from]]-Va[t,busIdx[lines[l].to]])-Yim*sin(Va[t,busIdx[lines[l].from]]-Va[t,busIdx[lines[l].to]])) 
                    ) 
                    - flowmax
                    <=0
                )
  
                #branch apparent power limits (to bus)
                Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
                Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
                @NLconstraint(opfmodel, 
                    Vm[t,busIdx[lines[l].to]]^2 *
                    ( Ytf_abs2*Vm[t,busIdx[lines[l].from]]^2 + Ytt_abs2*Vm[t,busIdx[lines[l].to]]^2
                    + 2*Vm[t,busIdx[lines[l].from]]*Vm[t,busIdx[lines[l].to]]*(Yre*cos(Va[t,busIdx[lines[l].from]]-Va[t,busIdx[lines[l].to]])-Yim*sin(Va[t,busIdx[lines[l].from]]-Va[t,busIdx[lines[l].to]]))
                    )
                    - flowmax
                    <=0
                )
            end
        end

        #@printf("Contingency %d -> Buses: %d  Lines: %d  Generators: %d\n", c, nbus, nline, ngen)
        #println("     lines with limits:  ", nlinelim)
    end
    return opfmodel
end

function solve_scmodel(opfmodel::JuMP.Model, opfdata::OPFData, rawdata::RawData; sc::Bool)
    T = sc ? (length(rawdata.ctgs_arr) + 1) : size(opfdata.Pd, 2)
    nbus = length(opfdata.buses)
    ngen = length(opfdata.generators)

    #
    # Warm-start all primal variables
    #
    init_x(opfmodel, opfdata, T; sc = sc)

    #
    # solve model
    #    
    status = solve(opfmodel)
    if status != :Optimal
        println("Stat is not optimal: ", status)
        return
    end


    #
    # Optimal primal solution
    #
    primal = initializePrimalSolution(opfdata, T; sc = sc)
    for t=1:T
        for g=1:ngen
            primal.PG[t,g] = getvalue(opfmodel[:Pg][t,g])
            primal.QG[t,g] = getvalue(opfmodel[:Qg][t,g])
            primal.SL[t,g] = getvalue(opfmodel[:Sl][t,g])
        end
        for b=1:nbus
            primal.VM[t,b] = getvalue(opfmodel[:Vm][t,b])
            primal.VA[t,b] = getvalue(opfmodel[:Va][t,b])
        end
    end


    #
    # Optimal dual solution
    #
    dual = initializeDualSolution(opfdata, T; sc = sc)
    for t=2:T
        for g=1:ngen
            dual.λ[t,g] = internalmodel(opfmodel).inner.mult_g[linearindex(opfmodel[:ramping][t,g])]
        end
    end


    return primal, dual
end

function scopf_model(opfdata::OPFData, rawdata::RawData, t::Int; sc::Bool, params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)

    # the model
    opfmodel = Model(solver=IpoptSolver(print_level=1))


    #shortcuts for compactness
    buses = opfdata.buses
    baseMVA = opfdata.baseMVA
    generators = opfdata.generators
    ngen  = length(generators)
    nbus  = length(buses)

    @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax)
    @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax)
    @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus] <= buses[i].Vmax)
    @variable(opfmodel, Va[1:nbus])

    # slack variable for ramping constraints
    @variable(opfmodel, 0 <= Sl[g=1:ngen] <= 2Float64(t > 1)*(sc ? generators[g].scen_agc : generators[g].ramp_agc))

    # reference bus voltage angle
    setlowerbound(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)
    setupperbound(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)

    #
    # Penalty terms
    #
    penalty = penalty_expression(opfmodel, opfdata, t; sc = sc, params = params, dual = dual, primal = primal)

    # objective function
    if sc && t > 1
        @objective(opfmodel, Min, penalty)
    else    
        @objective(opfmodel, Min, penalty +
            sum(  generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[g])^2
                + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[g])
                + generators[g].coeff[generators[g].n  ] for g=1:ngen)
        )
    end



    # Network data
    opfdata_c = opfdata

    # Perturbed network data for contingencies
    if sc
        if t > 1
            opfdata_c = opf_loaddata(rawdata; lineOff = opfdata.lines[rawdata.ctgs_arr[t - 1]])
        end
        Pd = [buses[b].Pd for b in 1:nbus]
        Qd = [buses[b].Qd for b in 1:nbus]
    else
        Pd = opfdata.Pd[:,t]
        Qd = opfdata.Qd[:,t]
    end

    # Network data short-hands
    lines = opfdata_c.lines
    busIdx = opfdata_c.BusIdx
    FromLines = opfdata_c.FromLines
    ToLines = opfdata_c.ToLines
    BusGeners = opfdata_c.BusGenerators
    nline = length(lines)
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
        

    # Power Balance Equations
    for b in 1:nbus
        #real part
        @NLconstraint(opfmodel, 
            ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b]^2 
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *( YftR[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )  
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*( YtfR[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfI[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   ) 
            - ( sum(baseMVA*Pg[g] for g in BusGeners[b]) - Pd[b]) / baseMVA      # Sbus part
            ==0
        )

        #imaginary part
        @NLconstraint(opfmodel,
            ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b]^2 
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
            - ( sum(baseMVA*Qg[g] for g in BusGeners[b]) - Qd[b]) / baseMVA      #Sbus part
            ==0
        )
    end


    # Line flow limits
    nlinelim=0
    for l in 1:nline
        if lines[l].rateA!=0 && lines[l].rateA<1.0e10
            nlinelim += 1
            flowmax=(lines[l].rateA/baseMVA)^2

            #branch apparent power limits (from bus)
            Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
            Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
            @NLconstraint(opfmodel,
                Vm[busIdx[lines[l].from]]^2 *
                ( Yff_abs2*Vm[busIdx[lines[l].from]]^2 + Yft_abs2*Vm[busIdx[lines[l].to]]^2 
                + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])) 
                ) 
                - flowmax
                <=0
            )

            #branch apparent power limits (to bus)
            Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
            Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
            @NLconstraint(opfmodel, 
                Vm[busIdx[lines[l].to]]^2 *
                ( Ytf_abs2*Vm[busIdx[lines[l].from]]^2 + Ytt_abs2*Vm[busIdx[lines[l].to]]^2
                + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]]))
                )
                - flowmax
                <=0
            )
        end
    end

    return opfmodel
end

function penalty_expression(opfmodel::JuMP.Model, opfdata::OPFData, t::Int; sc::Bool, params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)
    Pg = opfmodel[:Pg]
    Qg = opfmodel[:Qg]
    Vm = opfmodel[:Vm]
    Va = opfmodel[:Va]
    Sl = opfmodel[:Sl]
    gen = opfdata.generators

    #
    # Penalty terms for ALADIN
    #
    @expression(opfmodel, penalty, 0)
    if (params.aladin)
        error("to do: ALADIN for mpacopf")
    #
    # Penalty terms for prox-ALM
    #
    else
        for g=1:length(gen)
            # security-constrained opf
            if sc
                if t > 1
                    penalty += dual.λ[t,g]*(+primal.PG[1,g] - Pg[g] + Sl[g] - gen[g].scen_agc)
                    penalty += 0.5params.ρ*(+primal.PG[1,g] - Pg[g] + Sl[g] - gen[g].scen_agc)^2
                else
                    for s=2:size(dual.λ, 1)
                        penalty += dual.λ[s,g]*(+Pg[g] - primal.PG[s,g] + primal.SL[s,g] - gen[g].scen_agc)
                        penalty += 0.5params.ρ*(+Pg[g] - primal.PG[s,g] + primal.SL[s,g] - gen[g].scen_agc)^2
                    end
                end

            # multi-period opf
            else
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
    end

    #
    # the proximal part
    #
    if !iszero(params.τ)
        for g=1:length(gen)
            penalty += 0.5*params.τ*(Pg[g] - primal.PG[t,g])^2
        end
    end

    return penalty
end

function solve_scmodel(opfmodel::JuMP.Model, opfdata::OPFData, t::Int;
        sc::Bool, initial_x::mpPrimalSolution = nothing, initial_λ::mpDualSolution = nothing, params::AlgParams)
    Pg = opfmodel[:Pg]
    Qg = opfmodel[:Qg]
    Vm = opfmodel[:Vm]
    Va = opfmodel[:Va]
    Sl = opfmodel[:Sl]
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
            xp = (initial_x == nothing) ? 0 : ((sc ? initial_x.PG[1,g] : initial_x.PG[t-1,g]) - initial_x.PG[t,g])
            dp = (sc ? gen[g].scen_agc : gen[g].ramp_agc)
            setvalue(Sl[g], min(max(0, -xp + dp - tp), 2dp))
        end
    end

    #
    # Solve model
    #
    status = solve(opfmodel)

    return opfmodel, status
end
