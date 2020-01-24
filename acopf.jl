include("opfdata.jl")
include("partition.jl")
include("params.jl")

using MathProgBase
using MathOptInterface
using JuMP
using Printf
using Ipopt

function acopf_solve(opfmodel, opfdata, network::OPFNetwork, partition_idx::Int)

    # 
    # Initial point
    #
    for g in network.gener_part[partition_idx]
        setvalue(opfmodel[:Pg][g], 0.5*(opfdata.generators[g].Pmax+opfdata.generators[g].Pmin))
        setvalue(opfmodel[:Qg][g], 0.5*(opfdata.generators[g].Qmax+opfdata.generators[g].Qmin))
    end
    for b in network.buses_bloc[partition_idx]
        setvalue(opfmodel[:Vm][b], 0.5*(opfdata.buses[b].Vmax+opfdata.buses[b].Vmin))
        setvalue(opfmodel[:Va][b], opfdata.buses[opfdata.bus_ref].Va)
    end

    # 
    # Solve model
    #
    status = solve(opfmodel)

    return opfmodel,status
end


function acopf_model(opfdata, network::OPFNetwork, params::ALADINParams, partition_idx::Int)
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
    #=
    @NLparameter(opfmodel, VMsrcT[i=1:network.num_consensus] == params.VMsrcT[i])
    @NLparameter(opfmodel, VMsrcC[i=1:network.num_consensus] == params.VMsrcC[i])
    @NLparameter(opfmodel, VMdstT[i=1:network.num_consensus] == params.VMdstT[i])
    @NLparameter(opfmodel, VMdstC[i=1:network.num_consensus] == params.VMdstC[i])
    @NLparameter(opfmodel, VAsrcT[i=1:network.num_consensus] == params.VAsrcT[i])
    @NLparameter(opfmodel, VAsrcC[i=1:network.num_consensus] == params.VAsrcC[i])
    @NLparameter(opfmodel, VAdstT[i=1:network.num_consensus] == params.VAdstT[i])
    @NLparameter(opfmodel, VAdstC[i=1:network.num_consensus] == params.VAdstC[i])
    @NLparameter(opfmodel, λVMsrc[i=1:network.num_consensus] == params.λVMsrc[i])
    @NLparameter(opfmodel, λVMdst[i=1:network.num_consensus] == params.λVMdst[i])
    @NLparameter(opfmodel, λVAsrc[i=1:network.num_consensus] == params.λVAsrc[i])
    @NLparameter(opfmodel, λVAdst[i=1:network.num_consensus] == params.λVAdst[i])
    =#
    @expression(opfmodel, penalty, 0)
    for b in buses_part
        (b in network.consensus_nodes) || continue
        for key in keys(params.λVM)
            (key[1] == b) || continue
            penalty += (params.λVM[key]*Vm[b]) + (0.5*params.ρ*(params.VM[partition_idx][b] - Vm[b])^2)
            penalty += (params.λVA[key]*Vm[b]) + (0.5*params.ρ*(params.VA[partition_idx][b] - Va[b])^2)
        end
        for j in neighbors(network.graph, b)
            (get_prop(network.graph, j, :partition) == partition_idx) && continue
            @assert haskey(params.λVM, (j, partition_idx))
            penalty += -(params.λVM[(j, partition_idx)]*Vm[j]) + (0.5*params.ρ*(params.VM[partition_idx][j] - Vm[j])^2)
            penalty += -(params.λVA[(j, partition_idx)]*Vm[j]) + (0.5*params.ρ*(params.VA[partition_idx][j] - Va[j])^2)
        end
    end

    #=
    for e in edges(network.graph)
        get_prop(network.graph, e, :consensus) || continue
        i = src(e)
        j = dst(e)
        psrc = get_prop(network.graph, i, :partition)
        pdst = get_prop(network.graph, j, :partition)
        if i in buses_part
            idx = get_prop(network.graph, e, :index)
            ρ = params.ρ
            penalty += +(params.λVMsrc[idx]*Vm[i]) + (0.5*ρ*(params.VMsrcT[idx] - Vm[i])^2)
            penalty += +(params.λVAsrc[idx]*Va[i]) + (0.5*ρ*(params.VAsrcT[idx] - Va[i])^2)
            penalty += -(params.λVMdst[idx]*Vm[j]) + (0.5*ρ*(params.VMdstC[idx] - Vm[j])^2)
            penalty += -(params.λVAdst[idx]*Va[j]) + (0.5*ρ*(params.VAdstC[idx] - Va[j])^2)
        elseif j in buses_part
            idx = get_prop(network.graph, e, :index)
            ρ = params.ρ
            penalty += -(params.λVMsrc[idx]*Vm[i]) + (0.5*ρ*(params.VM[pdst][i] - Vm[i])^2)
            penalty += -(params.λVAsrc[idx]*Va[i]) + (0.5*ρ*(params.VA[pdst][i] - Va[i])^2)
            penalty += +(params.λVMdst[idx]*Vm[j]) + (0.5*ρ*(params.VM[pdst][j] - Vm[j])^2)
            penalty += +(params.λVAdst[idx]*Va[j]) + (0.5*ρ*(params.VA[pdst][j] - Va[j])^2)
        end
    end
    =#


    #
    # Generation cost
    #
    @expression(opfmodel, cost, sum( generators[i].coeff[generators[i].n-2]*(baseMVA*Pg[i])^2 
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

