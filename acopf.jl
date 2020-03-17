#
# ACOPF model
#

function penalty_expression(opfmodel, opfdata, network::OPFNetwork, partition_idx::Int; params::AlgParams, dual::DualSolution, primal::PrimalSolution)
    Pg = opfmodel[:Pg]
    Qg = opfmodel[:Qg]
    Vm = opfmodel[:Vm]
    Va = opfmodel[:Va]

    #
    # Penalty terms for ALADIN
    #
    @expression(opfmodel, penalty, 0)
    if (params.aladin)
        for (b, p, q) in network.consensus_tuple
            (b in network.buses_bloc[partition_idx]) || continue
            (p != partition_idx && q != partition_idx) && continue

            key = (b, p, q)
            coef = (p == partition_idx) ? 1.0 : -1.0
            penalty += (coef*dual.λVM[key]*(Vm[b] - primal.VM[partition_idx][b])) +(0.5*params.ρ*(Vm[b] - primal.VM[partition_idx][b])^2)
            penalty += (coef*dual.λVA[key]*(Va[b] - primal.VA[partition_idx][b])) +(0.5*params.ρ*(Va[b] - primal.VA[partition_idx][b])^2) 
        end

    #
    # Penalty terms for prox-ALM
    #
    else
        for (b, p, q) in network.consensus_tuple
            (b in network.buses_bloc[partition_idx]) || continue
            (p != partition_idx && q != partition_idx) && continue

            key = (b, p, q)
            if p == partition_idx
                penalty += (dual.λVM[key]*(Vm[b] - primal.VM[q][b])) + ((0.5*params.ρ)*(Vm[b] - primal.VM[q][b])^2)
                penalty += (dual.λVA[key]*(Va[b] - primal.VA[q][b])) + ((0.5*params.ρ)*(Va[b] - primal.VA[q][b])^2)
            elseif q == partition_idx
                penalty += (dual.λVM[key]*(primal.VM[p][b] - Vm[b])) + ((0.5*params.ρ)*(primal.VM[p][b] - Vm[b])^2)
                penalty += (dual.λVA[key]*(primal.VA[p][b] - Va[b])) + ((0.5*params.ρ)*(primal.VA[p][b] - Va[b])^2)
            else
                @assert false
            end
        end
    end

    #
    # the proximal part
    #
    if !iszero(params.τ)
        for g in network.gener_part[partition_idx]
            penalty += 0.5*params.τ*(Pg[g] - primal.PG[partition_idx][g])^2
            penalty += 0.5*params.τ*(Qg[g] - primal.QG[partition_idx][g])^2
        end
        for b in network.buses_bloc[partition_idx]
            penalty += 0.5*params.τ*(Vm[b] - primal.VM[partition_idx][b])^2
            penalty += 0.5*params.τ*(Va[b] - primal.VA[partition_idx][b])^2
        end
    end

    return penalty
end


function acopf_solve(opfmodel, opfdata, network::OPFNetwork, partition_idx::Int; initial::PrimalSolution = nothing)

    #
    # Initial point
    #
    if initial == nothing
        for g in network.gener_part[partition_idx]
            setvalue(opfmodel[:Pg][g], 0.5*(opfdata.generators[g].Pmax+opfdata.generators[g].Pmin))
            setvalue(opfmodel[:Qg][g], 0.5*(opfdata.generators[g].Qmax+opfdata.generators[g].Qmin))
        end
        for b in network.buses_bloc[partition_idx]
            setvalue(opfmodel[:Vm][b], 0.5*(opfdata.buses[b].Vmax+opfdata.buses[b].Vmin))
            setvalue(opfmodel[:Va][b], opfdata.buses[opfdata.bus_ref].Va)
        end
    else
        for g in network.gener_part[partition_idx]
            setvalue(opfmodel[:Pg][g], initial.PG[partition_idx][g])
            setvalue(opfmodel[:Qg][g], initial.QG[partition_idx][g])
        end
        for b in network.buses_bloc[partition_idx]
            setvalue(opfmodel[:Vm][b], initial.VM[partition_idx][b])
            setvalue(opfmodel[:Va][b], initial.VA[partition_idx][b])
        end
    end

    #
    # Solve model
    #
    status = solve(opfmodel)

    return opfmodel,status
end


function acopf_model(opfdata, network::OPFNetwork, partition_idx::Int; params::AlgParams, dual::DualSolution, primal::PrimalSolution)
    #shortcuts for compactness
    lines = opfdata.lines;
    buses = opfdata.buses;
    generators = opfdata.generators;
    baseMVA = opfdata.baseMVA;
    busIdx = opfdata.BusIdx;
    FromLines = opfdata.FromLines;
    ToLines = opfdata.ToLines;
    BusGeners = opfdata.BusGenerators;
    nbus  = length(buses);
    nline = length(lines);
    ngen  = length(generators);

    @assert partition_idx >= 1 && partition_idx <= network.num_partitions
    buses_bloc = network.buses_bloc[partition_idx]
    buses_part = network.buses_part[partition_idx]
    gener_part = network.gener_part[partition_idx]

    #branch admitances
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)

    #
    # JuMP model
    #
    opfmodel = Model(solver = IpoptSolver(print_level = 1))

    @variable(opfmodel, generators[i].Pmin <= Pg[i in gener_part] <= generators[i].Pmax)
    @variable(opfmodel, generators[i].Qmin <= Qg[i in gener_part] <= generators[i].Qmax)
    @variable(opfmodel, buses[i].Vmin <= Vm[i in buses_bloc] <= buses[i].Vmax)
    @variable(opfmodel, -pi <= Va[i in buses_bloc] <= pi)

    #
    # fix the voltage angle at the reference bus
    #
    if opfdata.bus_ref in buses_bloc
        setlowerbound(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)
        setupperbound(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)
    end


    #
    # Penalty terms
    #
    penalty = penalty_expression(opfmodel, opfdata, network, partition_idx; params = params, dual = dual, primal = primal)


    #
    # Generation cost
    #
    @expression(opfmodel, cost, 0.001*sum( generators[i].coeff[generators[i].n-2]*(baseMVA*Pg[i])^2
        +generators[i].coeff[generators[i].n-1]*(baseMVA*Pg[i])
        +generators[i].coeff[generators[i].n  ] for i in gener_part))

    #
    # Objective
    #
    @objective(opfmodel, Min, cost + penalty)

    #
    # power flow balance
    #
    for b in buses_part
        #real part
        @NLconstraint(
            opfmodel,
            ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b]^2
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *( YftR[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*( YtfR[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfI[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
            - ( sum(baseMVA*Pg[g] for g in BusGeners[b]) - buses[b].Pd ) / baseMVA      # Sbus part
            ==0)
        #imaginary part
        @NLconstraint(
            opfmodel,
            ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b]^2
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
            - ( sum(baseMVA*Qg[g] for g in BusGeners[b]) - buses[b].Qd ) / baseMVA      #Sbus part
            ==0)
    end
    #
    # branch/lines flow limits
    #
    nlinelim=0
    for l in 1:nline
        if busIdx[lines[l].from] ∉ buses_bloc ||
            busIdx[lines[l].to] ∉ buses_bloc ||
            lines[l].rateA==0 ||
            lines[l].rateA>=1.0e10
            continue
        end
        nlinelim += 1
        flowmax=(lines[l].rateA/baseMVA)^2

        #branch apparent power limits (from bus)
        Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
        Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
        @NLconstraint(opfmodel,
            Vm[busIdx[lines[l].from]]^2 *
            (
                Yff_abs2*Vm[busIdx[lines[l].from]]^2 + Yft_abs2*Vm[busIdx[lines[l].to]]^2
                + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]]))
                )
            - flowmax <=0)

        #branch apparent power limits (to bus)
        Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
        Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
        @NLconstraint(opfmodel,
            Vm[busIdx[lines[l].to]]^2 *
            (
                Ytf_abs2*Vm[busIdx[lines[l].from]]^2 + Ytt_abs2*Vm[busIdx[lines[l].to]]^2
                + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]]))
                )
            - flowmax <=0)
    end


    return opfmodel
end


function acopf_outputAll(opfmodel, opfdata, network::OPFNetwork, partition_idx::Int)
    lines = opfdata.lines; buses = opfdata.buses; generators = opfdata.generators; baseMVA = opfdata.baseMVA
    busIdx = opfdata.BusIdx; FromLines = opfdata.FromLines; ToLines = opfdata.ToLines; BusGeners = opfdata.BusGenerators;
    nbus  = length(buses);
    nline = length(lines);
    ngen  = length(generators)

    # OUTPUTING
    objval = getobjectivevalue(opfmodel)
    println("Objective value: ", objval, "USD/hr")
    VM=getvalue(getindex(opfmodel,:Vm)); VA=getvalue(getindex(opfmodel,:Va))
    PG=getvalue(getindex(opfmodel,:Pg)); QG=getvalue(getindex(opfmodel,:Qg))

    @assert partition_idx >= 1 && partition_idx <= network.num_partitions

    println("============================= BUSES ==================================")
    println("  BUS    Vm     Va    |  Pg (MW)    Qg(MVAr) ")
    println("----------------------------------------------------------------------")
    for i in network.buses_part[partition_idx]
        @printf("%4d | %6.2f  %6.2f | %s  | \n",
            buses[i].bus_i, VM[i], VA[i]*180/pi,
            length(BusGeners[i])==0 ? "   --          --  " :
            @sprintf("%7.2f     %7.2f", baseMVA*PG[BusGeners[i][1]], baseMVA*QG[BusGeners[i][1]]))
    end
    println("\n")
    return
end


function acopf_initialPt_IPOPT(opfdata, network::OPFNetwork, partition_idx::Int)
    gen = opfdata.generators
    Pg=zeros(length(network.gener_part[partition_idx]));
    Qg=zeros(length(network.gener_part[partition_idx]));
    for (i, g) in enumerate(network.gener_part[partition_idx])
        Pg[i]=0.5*(gen[g].Pmax+gen[g].Pmin)
        Qg[i]=0.5*(gen[g].Qmax+gen[g].Qmin)
    end

    Vm=zeros(length(network.buses_bloc[partition_idx]));
    for (i, b) in enumerate(network.buses_bloc[partition_idx])
        Vm[i]=0.5*(opfdata.buses[b].Vmax+opfdata.buses[b].Vmin);
    end

    # set all angles to the angle of the reference bus
    Va = opfdata.buses[opfdata.bus_ref].Va * ones(length(network.buses_bloc[partition_idx]))

    return Pg,Qg,Vm,Va
end


#################################################################################
#################################################################################

function acopf_solve_monolithic(opfmodel, opfdata, network::OPFNetwork)

    #
    # Initial point
    #
    for p in 1:network.num_partitions
        for g in network.gener_part[p]
            setvalue(opfmodel[:PG][p,g], 0.5*(opfdata.generators[g].Pmax+opfdata.generators[g].Pmin))
            setvalue(opfmodel[:QG][p,g], 0.5*(opfdata.generators[g].Qmax+opfdata.generators[g].Qmin))
        end
        for b in network.buses_bloc[p]
            setvalue(opfmodel[:VM][p,b], 0.5*(opfdata.buses[b].Vmax+opfdata.buses[b].Vmin))
            setvalue(opfmodel[:VA][p,b], opfdata.buses[opfdata.bus_ref].Va)
        end
    end

    #
    # Solve model
    #
    status = solve(opfmodel)
    if status != :Optimal
        error("something went wrong when solving monolithic model with status ", status)
    end


    #
    # Optimal primal solution
    #
    primal = initializePrimalSolution(opfdata, network)
    for p in 1:network.num_partitions
        for b in network.buses_bloc[p]
            primal.VM[p][b] = getvalue(opfmodel[:VM][p,b])
            primal.VA[p][b] = getvalue(opfmodel[:VA][p,b])
        end
        for g in network.gener_part[p]
            primal.PG[p][g] = getvalue(opfmodel[:PG][p,g])
            primal.QG[p][g] = getvalue(opfmodel[:QG][p,g])
        end
    end


    #
    # Optimal dual solution
    #
    dual = initializeDualSolution(opfdata, network)
    for key in network.consensus_tuple
        dual.λVM[key] = internalmodel(opfmodel).inner.mult_g[linearindex(opfmodel[:coupling_vm][key])]
        dual.λVA[key] = internalmodel(opfmodel).inner.mult_g[linearindex(opfmodel[:coupling_vm][key])]
    end


    return primal, dual
end


function acopf_model_monolithic(opfdata, network::OPFNetwork)
    #shortcuts for compactness
    lines = opfdata.lines;
    buses = opfdata.buses;
    generators = opfdata.generators;
    baseMVA = opfdata.baseMVA;
    busIdx = opfdata.BusIdx;
    FromLines = opfdata.FromLines;
    ToLines = opfdata.ToLines;
    BusGeners = opfdata.BusGenerators;
    nbus  = length(buses);
    nline = length(lines);
    ngen  = length(generators);

    #branch admitances
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)

    #
    # JuMP model
    #
    opfmodel = Model(solver = IpoptSolver(print_level = 1))

    @variable(opfmodel, generators[i].Pmin <= PG[p = 1:network.num_partitions, i in network.gener_part[p]] <= generators[i].Pmax)
    @variable(opfmodel, generators[i].Qmin <= QG[p = 1:network.num_partitions, i in network.gener_part[p]] <= generators[i].Qmax)
    @variable(opfmodel, buses[i].Vmin <= VM[p = 1:network.num_partitions, i in network.buses_bloc[p]] <= buses[i].Vmax)
    @variable(opfmodel, -pi <= VA[p = 1:network.num_partitions, i in network.buses_bloc[p]] <= pi)

    #
    # fix the voltage angle at the reference bus
    #
    for p in 1:network.num_partitions
        if opfdata.bus_ref in network.buses_bloc[p]
            setlowerbound(VA[p,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
            setupperbound(VA[p,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
        end
    end


    #
    # Add consensus constraints
    #
    @NLconstraint(opfmodel, coupling_vm[key in network.consensus_tuple], VM[key[2],key[1]] == VM[key[3],key[1]])
    @NLconstraint(opfmodel, coupling_va[key in network.consensus_tuple], VA[key[2],key[1]] == VA[key[3],key[1]])


    #
    # Generation cost
    #
    @objective(opfmodel, Min, 0.001*sum(sum( generators[i].coeff[generators[i].n-2]*(baseMVA*PG[p,i])^2
        +generators[i].coeff[generators[i].n-1]*(baseMVA*PG[p,i])
        +generators[i].coeff[generators[i].n  ] for i in network.gener_part[p]) for p in 1:network.num_partitions))


    #
    # power flow balance
    #
    for p in 1:network.num_partitions
        for b in network.buses_part[p]
            #real part
            @NLconstraint(
                opfmodel,
                ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * VM[p,b]^2
                + sum( VM[p,b]*VM[p,busIdx[lines[l].to]]  *( YftR[l]*cos(VA[p,b]-VA[p,busIdx[lines[l].to]]  ) + YftI[l]*sin(VA[p,b]-VA[p,busIdx[lines[l].to]]  )) for l in FromLines[b] )
                + sum( VM[p,b]*VM[p,busIdx[lines[l].from]]*( YtfR[l]*cos(VA[p,b]-VA[p,busIdx[lines[l].from]]) + YtfI[l]*sin(VA[p,b]-VA[p,busIdx[lines[l].from]])) for l in ToLines[b]   )
                - ( sum(baseMVA*PG[p,g] for g in BusGeners[b]) - buses[b].Pd ) / baseMVA      # Sbus part
                ==0)
            #imaginary part
            @NLconstraint(
                opfmodel,
                ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * VM[p,b]^2
                + sum( VM[p,b]*VM[p,busIdx[lines[l].to]]  *(-YftI[l]*cos(VA[p,b]-VA[p,busIdx[lines[l].to]]  ) + YftR[l]*sin(VA[p,b]-VA[p,busIdx[lines[l].to]]  )) for l in FromLines[b] )
                + sum( VM[p,b]*VM[p,busIdx[lines[l].from]]*(-YtfI[l]*cos(VA[p,b]-VA[p,busIdx[lines[l].from]]) + YtfR[l]*sin(VA[p,b]-VA[p,busIdx[lines[l].from]])) for l in ToLines[b]   )
                - ( sum(baseMVA*QG[p,g] for g in BusGeners[b]) - buses[b].Qd ) / baseMVA      #Sbus part
                ==0)
        end
    end
    #
    # branch/lines flow limits
    #
    nlinelim=0
    for p in 1:network.num_partitions
        for l in 1:nline
            if busIdx[lines[l].from] ∉ network.buses_bloc[p] ||
                busIdx[lines[l].to] ∉ network.buses_bloc[p] ||
                lines[l].rateA==0 ||
                lines[l].rateA>=1.0e10
                continue
            end
            nlinelim += 1
            flowmax=(lines[l].rateA/baseMVA)^2

            #branch apparent power limits (from bus)
            Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
            Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
            @NLconstraint(opfmodel,
                VM[p,busIdx[lines[l].from]]^2 *
                (
                    Yff_abs2*VM[p,busIdx[lines[l].from]]^2 + Yft_abs2*VM[p,busIdx[lines[l].to]]^2
                    + 2*VM[p,busIdx[lines[l].from]]*VM[p,busIdx[lines[l].to]]*(Yre*cos(VA[p,busIdx[lines[l].from]]-VA[p,busIdx[lines[l].to]])-Yim*sin(VA[p,busIdx[lines[l].from]]-VA[p,busIdx[lines[l].to]]))
                    )
                - flowmax <=0)

            #branch apparent power limits (to bus)
            Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
            Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
            @NLconstraint(opfmodel,
                VM[p,busIdx[lines[l].to]]^2 *
                (
                    Ytf_abs2*VM[p,busIdx[lines[l].from]]^2 + Ytt_abs2*VM[p,busIdx[lines[l].to]]^2
                    + 2*VM[p,busIdx[lines[l].from]]*VM[p,busIdx[lines[l].to]]*(Yre*cos(VA[p,busIdx[lines[l].from]]-VA[p,busIdx[lines[l].to]])-Yim*sin(VA[p,busIdx[lines[l].from]]-VA[p,busIdx[lines[l].to]]))
                    )
                - flowmax <=0)
        end
    end


    return opfmodel
end
