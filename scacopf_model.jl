using JuMP
using Ipopt

function init_x(opfmodel::JuMP.Model, opfdata::OPFData, T::Int)
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
end

function scopf_model(opfdata::OPFData, rawdata::RawData; options::Option = Option())

    # number of variable blocks
    T = 1

    # multi-period OPF
    if options.has_ramping
        T = size(opfdata.Pd, 2)
    end

    lines_off = [Line() for t in 1:T]

    # security-constrained OPF: explicitly add the "base case"
    if options.sc_constr
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

    # reference bus voltage angle
    for t in 1:T
        setlowerbound(Va[t,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
        setupperbound(Va[t,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
    end


    # Generation cost
    if options.obj_gencost
        @NLexpression(opfmodel, obj_gencost,
            sum((t > 1 ? options.weight_scencost : 1.0)*(
                    generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[1,g])^2
                    + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[1,g])
                    + generators[g].coeff[generators[g].n  ])
                for t=1:T, g=1:ngen)
        )
    else
        @NLexpression(opfmodel, obj_gencost, 0)
    end


    # slack variables to allow infeasibility in power flow equations
    if options.obj_penalty
        @variable(opfmodel, sigma_P1[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_P2[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_Q1[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_Q2[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_lineFrom[1:T,1:length(opfdata.lines)] >= 0, start = 0)
        @variable(opfmodel, sigma_lineTo[1:T,1:length(opfdata.lines)] >= 0, start = 0)


        @NLexpression(opfmodel, obj_penalty,
            sum(  sigma_P1[1,b] + sigma_P2[1,b] + sigma_Q1[1,b] + sigma_Q2[1,b] for b=1:nbus) +
            sum(  sigma_lineFrom[1,l] + sigma_lineTo[1,l] for l=1:length(opfdata.lines)) +
            (options.weight_scencost*
            sum(  sigma_P1[t,b] + sigma_P2[t,b] + sigma_Q1[t,b] + sigma_Q2[t,b] for t=2:T, b=1:nbus) +
            sum(  sigma_lineFrom[t,l] + sigma_lineTo[t,l] for t=2:T, l=1:length(opfdata.lines)))
        )
    else
        @NLexpression(opfmodel, sigma_P1[1:T,1:nbus], 0)
        @NLexpression(opfmodel, sigma_P2[1:T,1:nbus], 0)
        @NLexpression(opfmodel, sigma_Q1[1:T,1:nbus], 0)
        @NLexpression(opfmodel, sigma_Q2[1:T,1:nbus], 0)
        @NLexpression(opfmodel, sigma_lineFrom[1:T,1:length(opfdata.lines)], 0)
        @NLexpression(opfmodel, sigma_lineTo[1:T,1:length(opfdata.lines)], 0)


        @NLexpression(opfmodel, obj_penalty, 0)
    end


    # security-constrained opf
    if options.sc_constr
        #
        # Frequency control
        #
        if options.freq_ctrl
            # primary frequency control
            @variable(opfmodel, -1 <= omega[t=2:T] <= 1, start=0)
            @NLexpression(opfmodel, obj_freq_ctrl, 0.5*sum(omega[t]^2 for t=2:T))
        else
            # ignore
            @NLexpression(opfmodel, obj_freq_ctrl, 0)
        end

        #
        # Add coupling constraints
        #
        if options.two_block
            # 1st-stage variables
            @variable(opfmodel, generators[i].Pmin <= Pg_ref[i=1:ngen] <= generators[i].Pmax)

            # Copies of base-case power generation
            @variable(opfmodel, generators[i].Pmin <= Pg_base[t=1:T,i=1:ngen] <= generators[i].Pmax)

            # definition of Pg_base
            @constraint(opfmodel, [g=1:ngen],  Pg_base[1,g] - Pg[1,g] == 0)

            # consensus constraints
            @constraint(opfmodel, ramping_p[t=1:T,g=1:ngen], Pg_base[t,g] == Pg_ref[g])

            #
            # Reformulated coupling constraints -- no longer coupling
            #
            if options.freq_ctrl
                @constraint(opfmodel, [t=2:T,g=1:ngen],  Pg_base[t,g] - Pg[t,g] + (generators[g].alpha*omega[t]) == 0)
            else
                @constraint(opfmodel, [t=2:T,g=1:ngen],  Pg_base[t,g] - Pg[t,g] <= generators[g].scen_agc)
                @constraint(opfmodel, [t=2:T,g=1:ngen], -Pg_base[t,g] + Pg[t,g] <= generators[g].scen_agc)
            end
        else
            if options.freq_ctrl
                # primary frequency control under contingency
                @constraint(opfmodel, ramping_p[t=2:T,g=1:ngen],  Pg[1,g] - Pg[t,g] + (generators[g].alpha*omega[t]) == 0)
            else
                # power generation@(contingency - base case) <= bounded by ramp factor
                @constraint(opfmodel, ramping_p[t=2:T,g=1:ngen],  Pg[1,g] - Pg[t,g] <= generators[g].scen_agc)
                @constraint(opfmodel, ramping_n[t=2:T,g=1:ngen], -Pg[1,g] + Pg[t,g] <= generators[g].scen_agc)
            end
        end
    
    # multi-period opf
    elseif options.has_ramping
        # Ramping up/down constraints
        @constraint(opfmodel, ramping_p[t=2:T,g=1:ngen],  Pg[t-1,g] - Pg[t,g] <= generators[g].ramp_agc)
        @constraint(opfmodel, ramping_n[t=2:T,g=1:ngen], -Pg[t-1,g] + Pg[t,g] <= generators[g].ramp_agc)

        # ignore
        @NLexpression(opfmodel, obj_freq_ctrl, 0)
    end

    #
    # set objective function
    #
    @NLobjective(opfmodel,Min, 1e-3*(obj_gencost +
                                    (options.weight_loadshed*obj_penalty) +
                                    (options.weight_freqctrl*obj_freq_ctrl))
    )


    # constraints for each block
    for t in 1:T
        # Network data
        opfdata_c = opfdata

        # Perturbed network data for contingencies
        if options.sc_constr
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
                - ( sum(baseMVA*Pg[t,g] for g in BusGeners[b]) - Pd[b] + sigma_P1[t,b] - sigma_P2[t,b]) / baseMVA      # Sbus part
                ==0
            )

            #imaginary part
            @NLconstraint(opfmodel,
                ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[t,b]^2 
                + sum( Vm[t,b]*Vm[t,busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[t,b]-Va[t,busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[t,b]-Va[t,busIdx[lines[l].to]]  )) for l in FromLines[b] )
                + sum( Vm[t,b]*Vm[t,busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[t,b]-Va[t,busIdx[lines[l].from]]) + YtfR[l]*sin(Va[t,b]-Va[t,busIdx[lines[l].from]])) for l in ToLines[b]   )
                - ( sum(baseMVA*Qg[t,g] for g in BusGeners[b]) - Qd[b] + sigma_Q1[t,b] - sigma_Q2[t,b]) / baseMVA      #Sbus part
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
                    - (sigma_lineFrom[t,l]/baseMVA)
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
                    - (sigma_lineTo[t,l]/baseMVA)
                    <=0
                )
            end
        end

        #@printf("Contingency %d -> Buses: %d  Lines: %d  Generators: %d\n", c, nbus, nline, ngen)
        #println("     lines with limits:  ", nlinelim)
    end
    return opfmodel
end

function solve_scmodel(opfmodel::JuMP.Model, opfdata::OPFData, rawdata::RawData; options::Option = Option())
    T = options.sc_constr ? (length(rawdata.ctgs_arr) + 1) : size(opfdata.Pd, 2)
    nbus = length(opfdata.buses)
    ngen = length(opfdata.generators)

    #
    # Warm-start all primal variables
    #
    init_x(opfmodel, opfdata, T)

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
    primal = initializePrimalSolution(opfdata, T; options = options)
    for t=1:T
        for g=1:ngen
            primal.PG[t,g] = getvalue(opfmodel[:Pg][t,g])
            primal.QG[t,g] = getvalue(opfmodel[:Qg][t,g])
            if t > 1
                if options.sc_constr
                    if options.freq_ctrl
                        primal.SL[t] = getvalue(opfmodel[:omega][t])
                    else
                        rhs = opfdata.generators[g].scen_agc
                        primal.SL[t,g] = min(2rhs, max(0, rhs - primal.PG[1,g] + primal.PG[t,g]))
                    end
                elseif options.has_ramping
                    rhs = opfdata.generators[g].ramp_agc
                    primal.SL[t,g] = min(2rhs, max(0, rhs - primal.PG[t-1,g] + primal.PG[t,g]))
                end
            end
        end
        for b=1:nbus
            primal.VM[t,b] = getvalue(opfmodel[:Vm][t,b])
            primal.VA[t,b] = getvalue(opfmodel[:Va][t,b])
        end
    end
    if options.sc_constr && options.two_block
        for t=1:T, g=1:ngen
            primal.PG_BASE[t,g] = getvalue(opfmodel[:Pg_base][t,g])
            if t == 1
                primal.PG_REF[g] = getvalue(opfmodel[:Pg_ref][g])
            end
        end
    end


    #
    # Optimal dual solution
    #
    dual = initializeDualSolution(opfdata, T; options = options)
    for t=2:T
        for g=1:ngen
            dual.λp[t,g] = internalmodel(opfmodel).inner.mult_g[linearindex(opfmodel[:ramping_p][t,g])]
            if options.sc_constr && (options.freq_ctrl || options.two_block)
                continue
            end
            dual.λp[t,g] = abs(dual.λp[t,g])
            dual.λn[t,g] = abs(internalmodel(opfmodel).inner.mult_g[linearindex(opfmodel[:ramping_n][t,g])])
        end
    end


    return primal, dual
end

function scopf_model(opfdata::OPFData, rawdata::RawData, t::Int;
    options::Option = Option(),
    params::AlgParams,
    dual::mpDualSolution,
    primal::mpPrimalSolution)

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

    #
    # slack variables to allow infeasibility in power flow equations
    #
    if options.obj_penalty
        @variable(opfmodel, sigma_P1[1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_P2[1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_Q1[1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_Q2[1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_lineFrom[1:length(opfdata.lines)] >= 0, start = 0)
        @variable(opfmodel, sigma_lineTo[1:length(opfdata.lines)] >= 0, start = 0)

        @expression(opfmodel, obj_penalty, (t > 1 ? options.weight_scencost : 1.0)*(
            sum(  sigma_P1[b] + sigma_P2[b] + sigma_Q1[b] + sigma_Q2[b] for b=1:nbus) +
            sum(  sigma_lineFrom[l] + sigma_lineTo[l] for l=1:length(opfdata.lines)))
        )
    else
        @NLexpression(opfmodel, sigma_P1[1:nbus], 0)
        @NLexpression(opfmodel, sigma_P2[1:nbus], 0)
        @NLexpression(opfmodel, sigma_Q1[1:nbus], 0)
        @NLexpression(opfmodel, sigma_Q2[1:nbus], 0)
        @NLexpression(opfmodel, sigma_lineFrom[1:length(opfdata.lines)], 0)
        @NLexpression(opfmodel, sigma_lineTo[1:length(opfdata.lines)], 0)

        @expression(opfmodel, obj_penalty, 0)
    end

    #
    # slack variable for ramping constraints
    #
    if options.sc_constr
        # 2-block reformulation
        if options.two_block
            @variable(opfmodel, generators[i].Pmin <= Pg_base[i=1:ngen] <= generators[i].Pmax)
        end

        if options.freq_ctrl
            @variable(opfmodel, -Float64(t > 1) <= Sl <= Float64(t > 1), start = 0)
            if options.two_block
                @constraint(opfmodel, [g=1:ngen], Pg_base[g] - Pg[g] + (generators[g].alpha*Sl) == 0)
            end
        else
            @variable(opfmodel, 0 <= Sl[g=1:ngen] <= 2Float64(t > 1)*generators[g].scen_agc)
            if options.two_block
                if t > 1
                    @constraint(opfmodel, [g=1:ngen], Pg_base[g] - Pg[g] + Sl[g] == generators[g].scen_agc)
                else
                    @constraint(opfmodel, [g=1:ngen], Pg_base[g] - Pg[g] == 0)
                end
            end
        end
    else
        @variable(opfmodel, 0 <= Sl[g=1:ngen] <= 2Float64(t > 1)*generators[g].ramp_agc)
    end

    # reference bus voltage angle
    setlowerbound(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)
    setupperbound(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)

    #
    # Penalty terms
    #
    penalty = penalty_expression(opfmodel, opfdata, t; options = options, params = params, dual = dual, primal = primal)

    #
    # Generation cost
    #
    @expression(opfmodel, obj_gencost, 0)
    if options.obj_gencost
        obj_gencost += (t > 1 ? options.weight_scencost : 1.0)*
                        sum(  generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[g])^2
                            + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[g])
                            + generators[g].coeff[generators[g].n  ] for g=1:ngen)
    end

    #
    # Frequency control
    #
    @expression(opfmodel, obj_freq_ctrl, 0)
    if options.sc_constr && options.freq_ctrl && t > 1
        obj_freq_ctrl += 0.5*Sl^2
    end

    #
    # set objective function
    #
    @objective(opfmodel, Min, penalty +
        (1e-3*(
            obj_gencost + (options.weight_loadshed*obj_penalty) + (options.weight_freqctrl*obj_freq_ctrl)
            )
        )
    )


    # Network data
    opfdata_c = opfdata

    # Perturbed network data for contingencies
    if options.sc_constr
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
            - ( sum(baseMVA*Pg[g] for g in BusGeners[b]) - Pd[b] + sigma_P1[b] - sigma_P2[b]) / baseMVA      # Sbus part
            ==0
        )

        #imaginary part
        @NLconstraint(opfmodel,
            ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b]^2 
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
            - ( sum(baseMVA*Qg[g] for g in BusGeners[b]) - Qd[b] + sigma_Q1[b] - sigma_Q2[b]) / baseMVA      #Sbus part
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
                - (sigma_lineFrom[l]/baseMVA)
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
                - (sigma_lineTo[l]/baseMVA)
                <=0
            )
        end
    end

    return opfmodel
end

function penalty_expression(opfmodel::JuMP.Model, opfdata::OPFData, t::Int;
    options::Option = Option(),
    params::AlgParams,
    dual::mpDualSolution,
    primal::mpPrimalSolution)
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
            # two-block reformulation of security-constrained opf
            if options.sc_constr && options.two_block
                Pg_base = opfmodel[:Pg_base]
                penalty +=     dual.λp[t,g]*(Pg_base[g] - primal.PG_REF[g])
                penalty += 0.5params.ρ[t,g]*(Pg_base[g] - primal.PG_REF[g])^2

            # classical security-constrained opf
            elseif options.sc_constr && !options.freq_ctrl
                if t > 1
                    penalty +=     dual.λp[t,g]*(+primal.PG[1,g] - Pg[g] - gen[g].scen_agc)
                    penalty +=     dual.λn[t,g]*(-primal.PG[1,g] + Pg[g] - gen[g].scen_agc)
                    penalty += 0.5params.ρ[t,g]*(+primal.PG[1,g] - Pg[g] + Sl[g] - gen[g].scen_agc)^2
                else
                    for s=2:size(dual.λp, 1)
                        penalty +=     dual.λp[s,g]*(+Pg[g] - primal.PG[s,g] - gen[g].scen_agc)
                        penalty +=     dual.λn[s,g]*(-Pg[g] + primal.PG[s,g] - gen[g].scen_agc)
                        penalty += 0.5params.ρ[s,g]*(+Pg[g] - primal.PG[s,g] + primal.SL[s,g] - gen[g].scen_agc)^2
                    end
                end
            
            # frequency control security-constrained opf
            elseif options.sc_constr && options.freq_ctrl
                if t > 1
                    penalty +=     dual.λp[t,g]*(+primal.PG[1,g] - Pg[g] + (gen[g].alpha*Sl))
                    penalty += 0.5params.ρ[t,g]*(+primal.PG[1,g] - Pg[g] + (gen[g].alpha*Sl))^2
                else
                    for s=2:size(dual.λp, 1)
                        penalty +=     dual.λp[s,g]*(+Pg[g] - primal.PG[s,g] + (gen[g].alpha*primal.SL[s]))
                        penalty += 0.5params.ρ[s,g]*(+Pg[g] - primal.PG[s,g] + (gen[g].alpha*primal.SL[s]))^2
                    end
                end

            # multi-period opf
            elseif options.has_ramping
                if t > 1
                    penalty +=     dual.λp[t,g]*(+primal.PG[t-1,g] - Pg[g] - gen[g].ramp_agc)
                    penalty +=     dual.λn[t,g]*(-primal.PG[t-1,g] + Pg[g] - gen[g].ramp_agc)
                    penalty += 0.5params.ρ[t,g]*(+primal.PG[t-1,g] - Pg[g] + Sl[g] - gen[g].ramp_agc)^2
                end
                if t < size(opfdata.Pd, 2)
                    penalty +=   dual.λp[t+1,g]*(+Pg[g] - primal.PG[t+1,g] - gen[g].ramp_agc)
                    penalty +=   dual.λn[t+1,g]*(-Pg[g] + primal.PG[t+1,g] - gen[g].ramp_agc)
                    penalty += 0.5params.ρ[t,g]*(+Pg[g] - primal.PG[t+1,g] + primal.SL[t+1,g] - gen[g].ramp_agc)^2
                end
            end
        end
    end

    #
    # the proximal part
    #
    if !iszero(params.τ)
        if options.sc_constr && options.two_block
            if t == 1
                for g=1:length(gen)
                    penalty += 0.5*params.τ*(Pg[g] - primal.PG[t,g])^2
                end
            else
                Pg_base = opfmodel[:Pg_base]
                for g=1:length(gen)
                    penalty += 0.5*params.τ*(Pg_base[g] - primal.PG_BASE[t,g])^2
                end
            end
        else
            for g=1:length(gen)
                penalty += 0.5*params.τ*(Pg[g] - primal.PG[t,g])^2
            end
            if params.jacobi && options.sc_constr && options.freq_ctrl && t > 1
                penalty += 0.5*params.τ*(Sl - primal.SL[t])^2
            end
        end
    end

    return penalty
end

function solve_scmodel(opfmodel::JuMP.Model, opfdata::OPFData, t::Int;
    options::Option = Option(),
    initial_x::mpPrimalSolution = nothing,
    initial_λ::mpDualSolution = nothing,
    params::AlgParams)
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
            if options.sc_constr && options.two_block
                setvalue(opfmodel[:Pg_base][g], 0.5*(gen[g].Pmax + gen[g].Pmin))
            end
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
        if options.sc_constr
            if options.freq_ctrl
                setvalue(Sl, initial_x.SL[t])
            end
            if options.two_block
                setvalue.(opfmodel[:Pg_base], initial_x.PG_BASE[t,:])
            end
        end
    end
    #
    # Initial point for slack variables
    #
    if !options.freq_ctrl && t > 1
        for g=1:length(gen)
            xp = (initial_x == nothing) ? 0 : ((options.sc_constr ? initial_x.PG[1,g] : initial_x.PG[t-1,g]) - initial_x.PG[t,g])
            dp = (options.sc_constr ? gen[g].scen_agc : gen[g].ramp_agc)
            setvalue(Sl[g], min(max(0, -xp + dp), 2dp))
        end
    end

    #
    # Solve model
    #
    status = solve(opfmodel)

    return opfmodel, status
end
