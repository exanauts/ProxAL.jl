
function opf_fullmodel(opfdata::OPFData, rawdata::RawData, T::Int; options::Option = Option(), maxρ::Float64 = 1.0, compute_Lstar::Bool = false)

    # the model
    opfmodel = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5))


    #shortcuts for compactness
    buses = opfdata.buses
    baseMVA = opfdata.baseMVA
    generators = opfdata.generators
    ngen  = length(generators)
    nbus  = length(buses)

    @variable(opfmodel, generators[i].Pmin <= Pg[1:T,i=1:ngen] <= generators[i].Pmax, start = 0.5*(generators[i].Pmax + generators[i].Pmin))
    @variable(opfmodel, generators[i].Qmin <= Qg[1:T,i=1:ngen] <= generators[i].Qmax, start = 0.5*(generators[i].Qmax + generators[i].Qmin))
    @variable(opfmodel, buses[i].Vmin <= Vm[1:T,i=1:nbus] <= buses[i].Vmax, start = 0.5*(buses[i].Vmax + buses[i].Vmin))
    @variable(opfmodel, Va[1:T,1:nbus], start = buses[opfdata.bus_ref].Va)

    # reference bus voltage angle
    for t in 1:T
        fix(Va[t,opfdata.bus_ref], buses[opfdata.bus_ref].Va)
    end

    #
    # Quadratic penalty for computing lower bound on lyapunov sequence
    #
    @expression(opfmodel, lyapunov_quadratic_penalty, 0)

    #
    # Quadratic penalty to satisfy assumptions of convergence
    #
    @expression(opfmodel, obj_quadratic_penalty, 0)

    # Generation cost
    if options.obj_gencost
        @expression(opfmodel, obj_gencost,
            sum((t > 1 ? options.weight_scencost : 1.0)*(
                    generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[t,g])^2
                  + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[t,g])
                  + generators[g].coeff[generators[g].n  ])
                for t=1:T, g=1:ngen)
        )
    else
        @expression(opfmodel, obj_gencost, 0)
    end


    # slack variables to allow infeasibility in power flow equations
    if options.obj_penalty
        @variable(opfmodel, sigma_P1[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_P2[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_Q1[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_Q2[1:T,1:nbus] >= 0, start = 0)
        @variable(opfmodel, sigma_lineFrom[1:T,1:length(opfdata.lines)] >= 0, start = 0)
        @variable(opfmodel, sigma_lineTo[1:T,1:length(opfdata.lines)] >= 0, start = 0)


        @expression(opfmodel, obj_penalty,
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


        @expression(opfmodel, obj_penalty, 0)
    end


    # security-constrained opf
    if options.sc_constr
        #
        # Frequency control
        #
        if options.freq_ctrl
            # primary frequency control
            @variable(opfmodel, -1 <= omega[t=2:T] <= 1, start=0)
            @expression(opfmodel, obj_freq_ctrl, 0.5*sum(omega[t]^2 for t=2:T))
        else
            # ignore
            @expression(opfmodel, obj_freq_ctrl, 0)
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

            # Add coupling constraints to the objective -- we will delete the constraints later
            if compute_Lstar
                lyapunov_quadratic_penalty +=
                    sum(#=0.5params.ρ[t,g]*=#
                        (Pg_base[t,g] - Pg_ref[g])^2 for t=1:T,g=1:ngen
                    )
            end

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

                # Add coupling constraints to the objective -- we will delete the constraints later
                if compute_Lstar
                    lyapunov_quadratic_penalty += 
                        sum(#=0.5params.ρ[t,g]*=#
                            (Pg[1,g] - Pg[t,g] + (generators[g].alpha*omega[t]))^2 for t=2:T,g=1:ngen
                        )
                end
            else
                # power generation@(contingency - base case) <= bounded by ramp factor
                @constraint(opfmodel, ramping_p[t=2:T,g=1:ngen],  Pg[1,g] - Pg[t,g] <= generators[g].scen_agc)
                @constraint(opfmodel, ramping_n[t=2:T,g=1:ngen], -Pg[1,g] + Pg[t,g] <= generators[g].scen_agc)

                # Add coupling constraints to the objective -- we will delete the constraints later
                if compute_Lstar
                    @variable(opfmodel, 0 <= Sl[2:T,g=1:ngen] <= 2generators[g].scen_agc, start = generators[g].scen_agc)
                    lyapunov_quadratic_penalty += 
                        sum(#=0.5params.ρ[t,g]*=#
                            (Pg[1,g] - Pg[t,g] + Sl[t,g] - generators[g].scen_agc)^2 for t=2:T,g=1:ngen
                        )
                end
            end
        end
    
    # multi-period opf
    elseif options.has_ramping
        # Ramping up/down constraints
        if options.quadratic_penalty
            # Slack for ramping constraints
            @variable(opfmodel, 0 <= Sl[2:T,g=1:ngen] <= 2generators[g].ramp_agc, start = generators[g].ramp_agc)

            # Slack to satisfy assumptions for convergence
            @variable(opfmodel, quad[2:T,g=1:ngen], start = 0)

            # the actual ramping constraint with slacks
            @constraint(opfmodel, ramping_p[t=2:T,g=1:ngen],  Pg[t-1,g] - Pg[t,g] + Sl[t,g] + quad[t,g] == generators[g].ramp_agc)

            # Add coupling constraints to the objective -- we will delete the constraints later
            if compute_Lstar
                lyapunov_quadratic_penalty +=
                    sum(#=0.5params.ρ[t,g]*=#
                        (Pg[t-1,g] - Pg[t,g] + Sl[t,g] + quad[t,g] - generators[g].ramp_agc)^2 for t=2:T,g=1:ngen
                    )
            end

            # drive quad to zero
            obj_quadratic_penalty += 0.5*sum(quad[t,g]^2 for t=2:T,g=1:ngen)
        else
            @constraint(opfmodel, ramping_p[t=2:T,g=1:ngen],  Pg[t-1,g] - Pg[t,g] <= generators[g].ramp_agc)
            @constraint(opfmodel, ramping_n[t=2:T,g=1:ngen], -Pg[t-1,g] + Pg[t,g] <= generators[g].ramp_agc)

            # Add coupling constraints to the objective -- we will delete the constraints later
            if compute_Lstar
                @variable(opfmodel, 0 <= Sl[2:T,g=1:ngen] <= 2generators[g].ramp_agc, start = generators[g].ramp_agc)
                lyapunov_quadratic_penalty +=
                    sum(#=0.5params.ρ[t,g]*=#
                        (Pg[1,g] - Pg[t,g] + Sl[t,g] - generators[g].ramp_agc)^2 for t=2:T,g=1:ngen
                    )
            end
        end

        # ignore
        @expression(opfmodel, obj_freq_ctrl, 0)
    end

    #
    # delete coupling constraints when computing Lstar
    #
    if compute_Lstar
        drop_zeros!(lyapunov_quadratic_penalty)
        delete.(opfmodel, ramping_p)
        if !(options.sc_constr && (options.freq_ctrl || options.two_block)) &&
            !(options.has_ramping && options.quadratic_penalty)
            delete.(opfmodel, ramping_n)
        end
    end

    #
    # set objective function
    #
    @objective(opfmodel,Min, 1e-3*(obj_gencost +
                                  (options.weight_loadshed*obj_penalty) +
                                  (options.weight_freqctrl*obj_freq_ctrl) +
                                  (options.weight_quadratic_penalty*obj_quadratic_penalty)) +
                            (0.5maxρ*lyapunov_quadratic_penalty)
    )


    # constraints for each block
    for t in 1:T
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

        # Power Balance Equations
        add_real_power_balance_constraints(opfmodel, opfdata_c, Pg[t,:], Pd, Vm[t,:], Va[t,:], sigma_P1[t,:], sigma_P2[t,:])
        add_imag_power_balance_constraints(opfmodel, opfdata_c, Qg[t,:], Qd, Vm[t,:], Va[t,:], sigma_Q1[t,:], sigma_Q2[t,:])

        # Line flow limits
        add_line_power_constraints(opfmodel, opfdata_c, Vm[t,:], Va[t,:], sigma_lineFrom[t,:], sigma_lineTo[t,:])
    end
    return opfmodel
end

function opf_solve_fullmodel(opfmodel::JuMP.Model, opfdata::OPFData, rawdata::RawData, T::Int; options::Option = Option())
    #
    # solve model
    #    
    optimize!(opfmodel)
    status = termination_status(opfmodel)
    if  status != MOI.OPTIMAL &&
        status != MOI.ALMOST_OPTIMAL &&
        status != MOI.LOCALLY_SOLVED &&
        status != MOI.ALMOST_LOCALLY_SOLVED
        println("Non-decomposed model status is not optimal: ", status)
        return
    end


    #
    # Optimal primal solution
    #
    primal = initializePrimalSolution(opfdata, T; options = options)
    nbus = length(opfdata.buses)
    ngen = length(opfdata.generators)
    for t=1:T
        for g=1:ngen
            primal.PG[t,g] = value(opfmodel[:Pg][t,g])
            primal.QG[t,g] = value(opfmodel[:Qg][t,g])
            if t > 1
                if options.sc_constr
                    if options.freq_ctrl
                        primal.SL[t] = value(opfmodel[:omega][t])
                    else
                        rhs = opfdata.generators[g].scen_agc
                        primal.SL[t,g] = min(2rhs, max(0, rhs - primal.PG[1,g] + primal.PG[t,g]))
                    end
                elseif options.has_ramping
                    if options.quadratic_penalty
                        primal.ZZ[t,g] = value(opfmodel[:quad][t,g])
                        primal.SL[t,g] = value(opfmodel[:Sl][t,g])
                    else
                        rhs = opfdata.generators[g].ramp_agc
                        primal.SL[t,g] = min(2rhs, max(0, rhs - primal.PG[t-1,g] + primal.PG[t,g]))
                    end
                end
            end
        end
        for b=1:nbus
            primal.VM[t,b] = value(opfmodel[:Vm][t,b])
            primal.VA[t,b] = value(opfmodel[:Va][t,b])
        end
    end
    if options.sc_constr && options.two_block
        for t=1:T, g=1:ngen
            primal.PB[t,g] = value(opfmodel[:Pg_base][t,g])
            if t == 1
                primal.PR[g] = value(opfmodel[:Pg_ref][g])
            end
        end
    end


    #
    # Optimal dual solution
    #
    dual = initializeDualSolution(opfdata, T; options = options)
    for t=1:T
        if t == 1 && !(options.sc_constr && options.two_block)
            continue
        end
        for g=1:ngen
            dual.λp[t,g] = JuMP.dual(opfmodel[:ramping_p][t,g])
            if options.sc_constr && (options.freq_ctrl || options.two_block)
                # coupling equality constraint
                continue
            end
            if options.has_ramping && options.quadratic_penalty
                # coupling equality constraint
                continue
            end
            dual.λp[t,g] = abs(dual.λp[t,g])
            dual.λn[t,g] = abs(JuMP.dual(opfmodel[:ramping_n][t,g]))
        end
    end


    return primal, dual
end

function opf_model(opfdata::OPFData, rawdata::RawData, t::Int; options::Option = Option())

    # the model
    opfmodel = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 1))


    #shortcuts for compactness
    buses = opfdata.buses
    baseMVA = opfdata.baseMVA
    generators = opfdata.generators
    ngen  = length(generators)
    nbus  = length(buses)

    @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax, start = 0.5*(generators[i].Pmax + generators[i].Pmin))
    @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax, start = 0.5*(generators[i].Qmax + generators[i].Qmin))
    @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus] <= buses[i].Vmax, start = 0.5*(buses[i].Vmax + buses[i].Vmin))
    @variable(opfmodel, Va[1:nbus], start = buses[opfdata.bus_ref].Va)

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
    else
        @NLexpression(opfmodel, sigma_P1[1:nbus], 0)
        @NLexpression(opfmodel, sigma_P2[1:nbus], 0)
        @NLexpression(opfmodel, sigma_Q1[1:nbus], 0)
        @NLexpression(opfmodel, sigma_Q2[1:nbus], 0)
        @NLexpression(opfmodel, sigma_lineFrom[1:length(opfdata.lines)], 0)
        @NLexpression(opfmodel, sigma_lineTo[1:length(opfdata.lines)], 0)
    end

    #
    # slack variable for ramping constraints
    #
    if options.sc_constr
        # 2-block reformulation
        if options.two_block
            @variable(opfmodel,
                generators[i].Pmin <= Pg_base[i=1:ngen] <= generators[i].Pmax,
                start = 0.5*(generators[i].Pmax + generators[i].Pmin)
            )
        end

        if options.freq_ctrl
            @variable(opfmodel, -Float64(t > 1) <= Sl <= Float64(t > 1), start = 0)
            if options.two_block
                @constraint(opfmodel, [g=1:ngen], Pg_base[g] - Pg[g] + (generators[g].alpha*Sl) == 0)
            end
        else
            @variable(opfmodel,
                0 <= Sl[g=1:ngen] <= 2Float64(t > 1)*generators[g].scen_agc,
                start = Float64(t > 1)*generators[g].scen_agc
            )
            if options.two_block
                if t > 1
                    @constraint(opfmodel, [g=1:ngen], Pg_base[g] - Pg[g] + Sl[g] == generators[g].scen_agc)
                else
                    @constraint(opfmodel, [g=1:ngen], Pg_base[g] - Pg[g] == 0)
                end
            end
        end
    else
        @variable(opfmodel,
            0 <= Sl[g=1:ngen] <= 2Float64(t > 1)*generators[g].ramp_agc,
            start = Float64(t > 1)*generators[g].ramp_agc
        )
    end

    # reference bus voltage angle
    fix(Va[opfdata.bus_ref], buses[opfdata.bus_ref].Va)

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

    # Power Balance Equations
    add_real_power_balance_constraints(opfmodel, opfdata_c, Pg, Pd, Vm, Va, sigma_P1, sigma_P2)
    add_imag_power_balance_constraints(opfmodel, opfdata_c, Qg, Qd, Vm, Va, sigma_Q1, sigma_Q2)

    # Line flow limits
    add_line_power_constraints(opfmodel, opfdata_c, Vm, Va, sigma_lineFrom, sigma_lineTo)

    return opfmodel
end

function opf_model_set_objective(opfmodel::JuMP.Model, opfdata::OPFData, t::Int;
        options::Option = Option(), params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)
    #shortcuts for compactness
    buses = opfdata.buses
    baseMVA = opfdata.baseMVA
    generators = opfdata.generators
    ngen  = length(generators)
    nbus  = length(buses)

    #
    # AL term
    #
    proxAL_expr = opf_model_get_proxAL_expr(opfmodel, opfdata, t; options = options, params = params, dual = dual, primal = primal)

    #
    # Penalties for infeasibility
    #
    if options.obj_penalty
        sigma_P1 = opfmodel[:sigma_P1]
        sigma_P2 = opfmodel[:sigma_P2]
        sigma_Q1 = opfmodel[:sigma_Q1]
        sigma_Q2 = opfmodel[:sigma_Q2]
        sigma_lineFrom = opfmodel[:sigma_lineFrom]
        sigma_lineTo = opfmodel[:sigma_lineTo]
        obj_penalty = @expression(opfmodel,
            (t > 1 ? options.weight_scencost : 1.0)*(
                sum(  sigma_P1[b] + sigma_P2[b] + sigma_Q1[b] + sigma_Q2[b] for b=1:nbus) +
                sum(  sigma_lineFrom[l] + sigma_lineTo[l] for l=1:length(opfdata.lines))
                )
            )
    else
        obj_penalty = @expression(opfmodel, 0)
    end

    #
    # Generation cost
    #
    if options.obj_gencost
        Pg = opfmodel[:Pg]
        obj_gencost = @expression(opfmodel,
            (t > 1 ? options.weight_scencost : 1.0)*
                sum(  generators[g].coeff[generators[g].n-2]*(baseMVA*Pg[g])^2
                    + generators[g].coeff[generators[g].n-1]*(baseMVA*Pg[g])
                    + generators[g].coeff[generators[g].n  ] for g=1:ngen)
        )
    else
        obj_gencost = @expression(opfmodel, 0)
    end

    #
    # Frequency control
    #
    if options.sc_constr && options.freq_ctrl && t > 1
        Sl = opfmodel[:Sl]
        obj_freq_ctrl = @expression(opfmodel, 0.5*Sl^2)
    else
        obj_freq_ctrl = @expression(opfmodel, 0)
    end


    #
    # set objective function
    #
    @objective(opfmodel, Min, proxAL_expr +
        (1e-3*(
            obj_gencost + (options.weight_loadshed*obj_penalty) + (options.weight_freqctrl*obj_freq_ctrl)
            )
        )
    )

    return
end

function opf_model_get_proxAL_expr(opfmodel::JuMP.Model, opfdata::OPFData, t::Int;
        options::Option = Option(), params::AlgParams, dual::mpDualSolution, primal::mpPrimalSolution)
    Pg = opfmodel[:Pg]
    Qg = opfmodel[:Qg]
    Vm = opfmodel[:Vm]
    Va = opfmodel[:Va]
    Sl = opfmodel[:Sl]
    gen = opfdata.generators

    #
    # Penalty terms for ALADIN
    #
    penalty = @expression(opfmodel, 0)
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
                penalty +=     dual.λp[t,g]*(Pg_base[g] - primal.PR[g])
                penalty += 0.5params.ρ[t,g]*(Pg_base[g] - primal.PR[g])^2

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
            elseif options.has_ramping && options.quadratic_penalty
                if t > 1
                    penalty +=     dual.λp[t,g]*(+primal.PG[t-1,g] - Pg[g] + Sl[g] + primal.ZZ[t,g] - gen[g].ramp_agc)
                    penalty += 0.5params.ρ[t,g]*(+primal.PG[t-1,g] - Pg[g] + Sl[g] + primal.ZZ[t,g] - gen[g].ramp_agc)^2
                end
                if t < size(opfdata.Pd, 2)
                    penalty +=     dual.λp[t+1,g]*(+Pg[g] - primal.PG[t+1,g] + primal.SL[t+1,g] + primal.ZZ[t+1,g] - gen[g].ramp_agc)
                    penalty += 0.5params.ρ[t+1,g]*(+Pg[g] - primal.PG[t+1,g] + primal.SL[t+1,g] + primal.ZZ[t+1,g] - gen[g].ramp_agc)^2
                end
            elseif options.has_ramping
                if t > 1
                    penalty +=     dual.λp[t,g]*(+primal.PG[t-1,g] - Pg[g] - gen[g].ramp_agc)
                    penalty +=     dual.λn[t,g]*(-primal.PG[t-1,g] + Pg[g] - gen[g].ramp_agc)
                    penalty += 0.5params.ρ[t,g]*(+primal.PG[t-1,g] - Pg[g] + Sl[g] - gen[g].ramp_agc)^2
                end
                if t < size(opfdata.Pd, 2)
                    penalty +=     dual.λp[t+1,g]*(+Pg[g] - primal.PG[t+1,g] - gen[g].ramp_agc)
                    penalty +=     dual.λn[t+1,g]*(-Pg[g] + primal.PG[t+1,g] - gen[g].ramp_agc)
                    penalty += 0.5params.ρ[t+1,g]*(+Pg[g] - primal.PG[t+1,g] + primal.SL[t+1,g] - gen[g].ramp_agc)^2
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
                    penalty += 0.5*params.τ*(Pg_base[g] - primal.PB[t,g])^2
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
    drop_zeros!(penalty)

    return penalty
end

function opf_solve_model(opfmodel::JuMP.Model, t::Int)
    #
    # Initialize from the previous solve
    #
    if has_values(opfmodel)
        set_start_value.(all_variables(opfmodel), value.(all_variables(opfmodel)))
    end

    #
    # Solve model
    #
    optimize!(opfmodel)
    status = termination_status(opfmodel)
    if  status != MOI.OPTIMAL &&
        status != MOI.ALMOST_OPTIMAL &&
        status != MOI.LOCALLY_SOLVED &&
        status != MOI.ALMOST_LOCALLY_SOLVED &&
        status != MOI.ITERATION_LIMIT
        println("something went wrong in subproblem ", t, " with status ", status)
    end
    if !has_values(opfmodel)
        error("no solution vector available in subproblem ", t)
    end


    #
    # Return optimal solution vector
    #
    all_vars = all_variables(opfmodel)
    x = zeros(num_variables(opfmodel))
    for var in all_vars
        x[JuMP.optimizer_index(var).value] = value(var)
    end

    return x
end

function opf_solve_model(opfmodel::JuMP.Model, opfdata::OPFData, t::Int;
                         options::Option = Option(), initial_x::mpPrimalSolution, initial_λ::mpDualSolution, params::AlgParams)
    #
    # Initialize from given solution
    #
    set_start_value.(opfmodel[:Pg], initial_x.PG[t,:])
    set_start_value.(opfmodel[:Qg], initial_x.QG[t,:])
    set_start_value.(opfmodel[:Vm], initial_x.VM[t,:])
    set_start_value.(opfmodel[:Va], initial_x.VA[t,:])
    if options.sc_constr
        if options.freq_ctrl
            set_start_value(opfmodel[:Sl], initial_x.SL[t])
        end
        if options.two_block
            set_start_value.(opfmodel[:Pg_base], initial_x.PB[t,:])
        end
    end
    if !options.freq_ctrl && !options.two_block && t > 1
        for g=1:length(opfdata.generators)
            xp = (options.sc_constr ? initial_x.PG[1,g] : initial_x.PG[t-1,g]) - initial_x.PG[t,g]
            dp = (options.sc_constr ? opfdata.generators[g].scen_agc : opfdata.generators[g].ramp_agc)
            set_start_value(opfmodel[:Sl][g], min(max(0, -xp + dp), 2dp))
        end
    end


    #
    # Solve model
    #
    optimize!(opfmodel)
    status = termination_status(opfmodel)
    if  status != MOI.OPTIMAL &&
        status != MOI.ALMOST_OPTIMAL &&
        status != MOI.LOCALLY_SOLVED &&
        status != MOI.ALMOST_LOCALLY_SOLVED &&
        status != MOI.ITERATION_LIMIT
        println("something went wrong in subproblem ", t, " with status ", status)
    end
    if !has_values(opfmodel)
        error("no solution vector available in subproblem ", t)
    end


    #
    # Return optimal solution vector
    #
    all_vars = all_variables(opfmodel)
    x = zeros(num_variables(opfmodel))
    for var in all_vars
        x[JuMP.optimizer_index(var).value] = value(var)
    end

    return x
end

function computeLyapunovFunction(x::mpPrimalSolution, dual::mpDualSolution, xprev::mpPrimalSolution, opfdata::OPFData, T::Int; options::Option = Option(), params::AlgParams)
    gen = opfdata.generators

    # compute the primal cost
    primalcost = computePrimalCost(x, opfdata; options = options)

    # compute the AL part
    penalty = 0
    for t=1:T, g=1:length(gen)
        # two-block reformulation of security-constrained opf
        if options.sc_constr && options.two_block
            penalty +=     dual.λp[t,g]*(x.PB[t,g] - x.PR[g])
            penalty += 0.5params.ρ[t,g]*(x.PB[t,g] - x.PR[g])^2

        # classical security-constrained opf
        elseif options.sc_constr && !options.freq_ctrl
            if t > 1
                penalty +=     dual.λp[t,g]*(+x.PG[1,g] - x.PG[t,g] - gen[g].scen_agc)
                penalty +=     dual.λn[t,g]*(-x.PG[1,g] + x.PG[t,g] - gen[g].scen_agc)
                penalty += 0.5params.ρ[t,g]*(+x.PG[1,g] - x.PG[t,g] + x.SL[t,g] - gen[g].scen_agc)^2
            end

        # frequency control security-constrained opf
        elseif options.sc_constr && options.freq_ctrl
            if t > 1
                penalty +=     dual.λp[t,g]*(+x.PG[1,g] - x.PG[t,g] + (gen[g].alpha*x.SL[t]))
                penalty += 0.5params.ρ[t,g]*(+x.PG[1,g] - x.PG[t,g] + (gen[g].alpha*x.SL[t]))^2
            end

        # multi-period opf
        elseif options.has_ramping && options.quadratic_penalty
            if t > 1
                penalty +=     dual.λp[t,g]*(+x.PG[t-1,g] - x.PG[t,g] + x.SL[t,g] + x.ZZ[t,g] - gen[g].ramp_agc)
                penalty += 0.5params.ρ[t,g]*(+x.PG[t-1,g] - x.PG[t,g] + x.SL[t,g] + x.ZZ[t,g] - gen[g].ramp_agc)^2
            end
        elseif options.has_ramping
            if t > 1
                penalty +=     dual.λp[t,g]*(+x.PG[t-1,g] - x.PG[t,g] - gen[g].ramp_agc)
                penalty +=     dual.λn[t,g]*(-x.PG[t-1,g] + x.PG[t,g] - gen[g].ramp_agc)
                penalty += 0.5params.ρ[t,g]*(+x.PG[t-1,g] - x.PG[t,g] + x.SL[t,g] - gen[g].ramp_agc)^2
            end
        end
    end

    # compute the proximal part
    proximal = 0
    if options.sc_constr && options.two_block
        for g=1:length(gen)
            proximal += 0.5*params.τ*(x.PG[1,g] - xprev.PG[1,g])^2
            for t=2:T
                proximal += 0.5*params.τ*(x.PB[t,g] - xprev.PB[t,g])^2
            end
        end
    else
        for t=1:T, g=1:length(gen)
            proximal += 0.5*params.τ*(x.PG[t,g] - xprev.PG[t,g])^2
        end
        if params.jacobi && options.sc_constr && options.freq_ctrl
            for t=2:T
                proximal += 0.5*params.τ*(x.SL[t] - xprev.SL[t])^2
            end
        end
    end
    # prox from PgRef
    if options.sc_constr && options.two_block
        for g=1:length(gen)
            proximal += 0.5*params.τ*(x.PR[g] - xprev.PR[g])^2
        end
    end
    # prox from quadratic penalty
    if options.has_ramping && options.quadratic_penalty
        for t=2:T, g=1:length(gen)
            proximal += 0.5*params.τ*(x.ZZ[t,g] - xprev.ZZ[t,g])^2
        end
    end

    return primalcost, penalty, (0.5*proximal)
end

function add_line_power_constraints(opfmodel::JuMP.Model, opfdata::OPFData, Vm, Va, sigma_lineFrom, sigma_lineTo)
    # Network data short-hands
    baseMVA = opfdata.baseMVA
    buses = opfdata.buses
    lines = opfdata.lines
    busIdx = opfdata.BusIdx
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)

    for l in 1:length(lines)
        line = lines[l]
        from = busIdx[line.from]
        to = busIdx[line.to]
        if line.rateA ≈ 0 || line.rateA >= 1.0e10
            continue
        end
        flowmax=(line.rateA/baseMVA)^2

        #branch apparent power limits (from bus)
        Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
        Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
        @NLconstraint(opfmodel,
            Vm[from]^2 *
            ( Yff_abs2*Vm[from]^2 + Yft_abs2*Vm[to]^2 
            + 2*Vm[from]*Vm[to]*(Yre*cos(Va[from]-Va[to])-Yim*sin(Va[from]-Va[to])) 
            ) 
            - flowmax
            - (sigma_lineFrom[l]/baseMVA)
            <=0
        )

        #branch apparent power limits (to bus)
        Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
        Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
        @NLconstraint(opfmodel, 
            Vm[to]^2 *
            ( Ytf_abs2*Vm[from]^2 + Ytt_abs2*Vm[to]^2
            + 2*Vm[from]*Vm[to]*(Yre*cos(Va[from]-Va[to])-Yim*sin(Va[from]-Va[to]))
            )
            - flowmax
            - (sigma_lineTo[l]/baseMVA)
            <=0
        )
    end
end

function add_real_power_balance_constraints(opfmodel::JuMP.Model, opfdata::OPFData, Pg, Pd, Vm, Va, sigma_P1, sigma_P2)
    # Network data short-hands
    baseMVA = opfdata.baseMVA
    buses = opfdata.buses
    lines = opfdata.lines
    busIdx = opfdata.BusIdx
    FromLines = opfdata.FromLines
    ToLines = opfdata.ToLines
    BusGeners = opfdata.BusGenerators
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
        

    # Power Balance Equations
    for b in 1:length(buses)
        #real part
        @NLconstraint(opfmodel, 
            ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b]^2 
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *( YftR[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )  
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*( YtfR[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfI[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   ) 
            - ( sum(baseMVA*Pg[g] for g in BusGeners[b]) - Pd[b] + sigma_P1[b] - sigma_P2[b]) / baseMVA      # Sbus part
            ==0
        )
    end
end

function add_imag_power_balance_constraints(opfmodel::JuMP.Model, opfdata::OPFData, Qg, Qd, Vm, Va, sigma_Q1, sigma_Q2)
    # Network data short-hands
    baseMVA = opfdata.baseMVA
    buses = opfdata.buses
    lines = opfdata.lines
    busIdx = opfdata.BusIdx
    FromLines = opfdata.FromLines
    ToLines = opfdata.ToLines
    BusGeners = opfdata.BusGenerators
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
        

    # Power Balance Equations
    for b in 1:length(buses)
        #imaginary part
        @NLconstraint(opfmodel,
            ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b]^2 
            + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
            + sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
            - ( sum(baseMVA*Qg[g] for g in BusGeners[b]) - Qd[b] + sigma_Q1[b] - sigma_Q2[b]) / baseMVA      #Sbus part
            ==0
        )
    end
end
