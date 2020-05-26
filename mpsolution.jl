#
# Data structures to represent primal and dual solutions
#

mutable struct mpPrimalSolution
    PB::Array{Float64,2} #[t,g] := (two_block decomposition) copy of Pg[1,g] in scenario t
    PR::Array{Float64,1} #[g]   := (two_block decomposition) consensus variables Pr[g] == Pb[t,g] for all t
    PG::Array{Float64,2} #[t,g] := Pg of gen g in scenario/time period t
    QG::Array{Float64,2} #[t,g] := Qg of gen g in scenario/time period t
    VM::Array{Float64,2} #[t,b] := Vm of bus b in scenario/time period t
    VA::Array{Float64,2} #[t,b] := Va of bus b in scenario/time period t
    SL::Array{Float64}   #[t,g] := (mp acopf)  Slack for Pg[t-1,g] - Pg[t,g] <= r
                         #[t,g] := (sc acopf)  Slack for Pg[1,g] - Pg[t,g] <= r
                         #[t]   := (freq_ctrl) Frequency change variable: Pg[t,g] == Pg[1,g] + alpha*Sl[t]
end

mutable struct mpDualSolution
    λp::Array{Float64,2} #[t,g] := (two_block) λ for the constraint Pr[g] == Pb[t,g]
                         #[t,g] := (mp acopf)  λ for the constraint Pg[t-1,g] - Pg[t,g] <= r
                         #[t,g] := (sc acopf)  λ for the constraint Pg[1,g] - Pg[t,g] <= r
                         #[t,g] := (freq_ctrl) λ for the constraint Pg[t,g] == Pg[1,g] + alpha*Sl[t]
    λn::Array{Float64,2} #[t,g] := (mp/sc acopf) λ for the constraint -(Pg[1,g] or Pg[t-1,g]) + Pg[t,g] <= r
end

function initializeDualSolution(opfdata::OPFData, T::Int; options::Option = Option())
    λp = zeros(T, length(opfdata.generators))
    λn = zeros(T, length(opfdata.generators))

    return mpDualSolution(λp, λn)
end

function initializePrimalSolution(opfdata::OPFData, T::Int; options::Option = Option())
    bus = opfdata.buses
    gen = opfdata.generators
    num_buses = length(bus)
    num_gens = length(gen)

    Vm = zeros(T, num_buses)
    for b=1:num_buses
        Vm[:,b] .= 0.5*(bus[b].Vmax + bus[b].Vmin)
    end
    Va = bus[opfdata.bus_ref].Va * ones(T, num_buses)

    Pg = zeros(T, num_gens)
    Qg = zeros(T, num_gens)
    for g=1:num_gens
        Pg[:,g] .= 0.5*(gen[g].Pmax + gen[g].Pmin)
        Qg[:,g] .= 0.5*(gen[g].Qmax + gen[g].Qmin)
    end

    Pg_base = deepcopy(Pg)
    Pg_ref = deepcopy(Pg[1,:])

    if options.sc_constr && options.freq_ctrl
        Sl = zeros(T)
    else
        Sl = zeros(T, num_gens)
        for g=1:num_gens
            Sl[2:T,g] .= (options.sc_constr ? gen[g].scen_agc : gen[g].ramp_agc)
        end
    end

    return mpPrimalSolution(Pg_base, Pg_ref, Pg, Qg, Vm, Va, Sl)
end

function perturb(primal::mpPrimalSolution, factor::Number)
    primal.PB *= (1.0 + factor)
    primal.PR *= (1.0 + factor)
    primal.PG *= (1.0 + factor)
    primal.QG *= (1.0 + factor)
    primal.VM *= (1.0 + factor)
    primal.VA *= (1.0 + factor)
    primal.SL *= (1.0 + factor)
end

function perturb(dual::mpDualSolution, factor::Number)
    dual.λp *= (1.0 + factor)
    dual.λn *= (1.0 + factor)
end

function shiftBackward(primal::mpPrimalSolution)
    for t=1:size(primal.PB, 1)-1
        primal.PG[t,:] .= primal.PG[t+1,:]
        primal.QG[t,:] .= primal.QG[t+1,:]
        primal.VM[t,:] .= primal.VM[t+1,:]
        primal.VA[t,:] .= primal.VA[t+1,:]
        if length(size(primal.SL)) > 1
            primal.SL[t,:] .= primal.SL[t+1,:]
        else
            primal.SL[t] = primal.SL[t+1]
        end
    end
end

function shiftBackward(dual::mpDualSolution)
    for t=1:size(dual.λp, 1)-1
        dual.λp[t,:] .= dual.λp[t+1,:]
        dual.λn[t,:] .= dual.λn[t+1,:]
    end
end

function computeDistance(x1::mpPrimalSolution, x2::mpPrimalSolution; options::Option = Option(), lnorm = 1)
    if options.sc_constr
        if options.two_block
            xd = x1.PR .- x2.PR
            #xd = [vcat(xd...); x1.PR - x2.PR]
        elseif options.freq_ctrl
            xd = [x1.PG[1,:] - x2.PG[1,:]; x1.SL - x2.SL]
        else
            xd = x1.PG[1,:] - x2.PG[1,:]
        end
    else
        xd = x1.PG .- x2.PG
        #=
        xd = [[ x1.PG[t,:] - x2.PG[t,:];
                x1.QG[t,:] - x2.QG[t,:];
                x1.VM[t,:] - x2.VM[t,:];
                x1.VA[t,:] - x2.VA[t,:]] for t=1:size(x1.PG, 1)]
        =#
    end
    return norm(vcat(xd...), lnorm)
end

function computeDistance(x1::mpDualSolution, x2::mpDualSolution; lnorm = 1)
    xd = [[x1.λp[t,:] - x2.λp[t,:];
           x1.λn[t,:] - x2.λn[t,:]] for t=1:size(x1.λ, 1)]
    return norm(vcat(xd...), lnorm)
end

function computeDualViolation(x::mpPrimalSolution, xprev::mpPrimalSolution, λ::mpDualSolution, λprev::mpDualSolution, nlpdata::AlgNLPData, opfdata::OPFData;
                              options::Option = Option(), params::AlgParams, lnorm = 1)
    gen = opfdata.generators
    dualviol = []
    for t = 1:length(nlpdata.nlpmodel)
        #
        # First get ∇_x Lagrangian
        #
        # inner = internalmodel(nlpmodel[t]).inner
        #=
        kkt = grad_Lagrangian(nlpmodel[t], inner.x, inner.mult_g)
        for j = 1:length(kkt)
            if (abs(nlpmodel[t].colLower[j] - nlpmodel[t].colUpper[j]) <= params.zero)
                kkt[j] = 0.0 # ignore
            elseif abs(inner.x[j] - nlpmodel[t].colLower[j]) <= params.zero
                (kkt[j] < 0) && (kkt[j] = 0.0)
            elseif abs(inner.x[j] - nlpmodel[t].colUpper[j]) <= params.zero
                (kkt[j] > 0) && (kkt[j] = 0.0)
            else
                (kkt[j] != 0.0) && (kkt[j] = 0.0)
            end
        end
        =#
        kkt = zeros(nlpdata.colCount)

        #
        # Now adjust it so that the final quantity represents the error in the KKT conditions
        #
        if params.aladin
            error("to do")

        else
            # The Aug Lag part in proximal ALM
            for g=1:length(gen)
                idx_pg = nlpdata.colIndex["Pg[" * string(g) * "]"]
                # two-block reformulation
                if options.sc_constr && options.two_block
                    idx_pg_base = nlpdata.colIndex["Pg_base[" * string(g) * "]"]
                    kkt[idx_pg_base] += λ.λp[t,g] - λprev.λp[t,g]
                    kkt[idx_pg_base] -= params.ρ[t,g]*(x.PB[t,g] - xprev.PR[g])

                # contingency
                elseif options.sc_constr && !options.freq_ctrl
                    idx_sl = nlpdata.colIndex["Sl[" * string(g) * "]"]
                    if t > 1
                        kkt[idx_pg] += -λ.λp[t,g]+λ.λn[t,g]
                        temp = - params.ρ[t,g]*(
                                    (params.jacobi ? xprev.PG[1,g] : x.PG[1,g]) -
                                        x.PG[t,g] + x.SL[t,g] - gen[g].scen_agc
                                    )
                        kkt[idx_pg] -= -λprev.λp[t,g]+λprev.λn[t,g]+temp
                        kkt[idx_sl] += temp
                    else
                        for s=2:size(λ.λp, 1)
                            kkt[idx_pg] += +λ.λp[s,g]-λ.λn[s,g]
                            kkt[idx_pg] -= +λprev.λp[s,g]-λprev.λn[s,g] + params.ρ[s,g]*(
                                            x.PG[1,g] - xprev.PG[s,g] + xprev.SL[s,g] -
                                            gen[g].scen_agc
                                        )
                        end
                    end

                # frequency control
                elseif options.sc_constr && options.freq_ctrl
                    idx_sl = nlpdata.colIndex["Sl"]
                    if t > 1
                        kkt[idx_pg] += -λ.λp[t,g]
                        kkt[idx_sl] -= -gen[g].alpha*λ.λp[t,g]
                        temp = - params.ρ[t,g]*(
                                    (params.jacobi ? xprev.PG[1,g] : x.PG[1,g]) -
                                        x.PG[t,g] + (gen[g].alpha * x.SL[t])
                                    )
                        kkt[idx_pg] -= -λprev.λp[t,g]+temp
                        kkt[idx_sl] += gen[g].alpha*(-λprev.λp[t,g]+temp)
                    else
                        for s=2:size(λ.λp, 1)
                            kkt[idx_pg] += +λ.λp[s,g]
                            kkt[idx_pg] -= +λprev.λp[s,g] + params.ρ[s,g]*(
                                            x.PG[1,g] - xprev.PG[s,g] +
                                            (gen[g].alpha*xprev.SL[s])
                                        )
                        end
                    end

                # multiperiod
                elseif options.has_ramping
                    idx_sl = nlpdata.colIndex["Sl[" * string(g) * "]"]
                    if t > 1
                        kkt[idx_pg] += -λ.λp[t,g]+λ.λn[t,g]
                        temp = - params.ρ[t,g]*(
                                    (params.jacobi ? xprev.PG[t-1,g] : x.PG[t-1,g]) -
                                        x.PG[t,g] + x.SL[t,g] - gen[g].ramp_agc
                                    )
                        kkt[idx_pg] -= -λprev.λp[t,g]+λprev.λn[t,g]+temp
                        kkt[idx_sl] += temp
                    end
                    if t < length(nlpdata.nlpmodel)
                        kkt[idx_pg] += +λ.λp[t+1,g]-λ.λn[t+1,g]
                        kkt[idx_pg] -= +λprev.λp[t+1,g]-λprev.λn[t+1,g] + params.ρ[t+1,g]*(
                                        x.PG[t,g] - xprev.PG[t+1,g] + xprev.SL[t+1,g] -
                                        gen[g].ramp_agc
                                    )
                    end
                end
            end
        end
        # The proximal part in both ALADIN and proximal ALM
        if options.sc_constr && options.two_block
            for g=1:length(gen)
                idx_pg_base = nlpdata.colIndex["Pg_base[" * string(g) * "]"]
                kkt[idx_pg_base] -= params.τ*(x.PB[t,g] - xprev.PB[t,g])
            end
        else
            for g=1:length(gen)
                idx_pg = nlpdata.colIndex["Pg[" * string(g) * "]"]
                kkt[idx_pg] -= params.τ*(x.PG[t,g] - xprev.PG[t,g])
            end
            if params.jacobi && options.sc_constr && options.freq_ctrl && t > 1
                idx_sl = nlpdata.colIndex["Sl"]
                kkt[idx_sl] -= params.τ*(x.SL[t] - xprev.SL[t])
            end
        end

        for j = 1:length(kkt)
            if (abs(nlpdata.colLower[j, t] - nlpdata.colUpper[j, t]) <= params.zero)
                kkt[j] = 0.0 # ignore
            elseif abs(nlpdata.colValue[j, t] - nlpdata.colLower[j, t]) <= params.zero
                kkt[j] = max(0, -kkt[j])
            elseif abs(nlpdata.colValue[j, t] - nlpdata.colUpper[j, t]) <= params.zero
                kkt[j] = max(0, +kkt[j])
            else
                kkt[j] = abs(kkt[j])
            end
        end
        dualviol = [dualviol; kkt]
    end

    #
    # Pg_ref in two_block decomposition
    #
    if options.sc_constr && options.two_block
        for g=1:length(gen)
            kkt_pg_ref = -params.τ*(x.PR[g] - xprev.PR[g])
            for t=1:size(λ.λp, 1)
                kkt_pg_ref += -λ.λp[t,g]+λprev.λp[t,g]+(params.ρ[t,g]*(
                                x.PB[t,g] - x.PR[g]
                            ))
            end
            dualviol = [dualviol; kkt_pg_ref]
        end
    end

    return norm(dualviol, lnorm), norm(dualviol, 1)/length(dualviol)
end

function computePrimalViolation(primal::mpPrimalSolution, opfdata::OPFData; options::Option = Option(), lnorm = 1)
    T = size(primal.PG, 1)
    gen = opfdata.generators
    num_gens = length(gen)

    #
    # primal violation
    #
    if options.sc_constr && options.two_block
        errp = [abs(primal.PB[t,g] - primal.PR[g]) for t=1:T for g=1:num_gens]
        errn = zeros(0)
    elseif options.sc_constr && !options.freq_ctrl
        errp = [max(+primal.PG[1,g] - primal.PG[t,g] - gen[g].scen_agc, 0) for t=2:T for g=1:num_gens]
        errn = [max(-primal.PG[1,g] + primal.PG[t,g] - gen[g].scen_agc, 0) for t=2:T for g=1:num_gens]
    elseif options.sc_constr && options.freq_ctrl
        errp = [abs(+primal.PG[1,g] - primal.PG[t,g] + (gen[g].alpha*primal.SL[t])) for t=2:T for g=1:num_gens]
        errn = zeros(0)
    elseif options.has_ramping
        errp = [max(+primal.PG[t-1,g] - primal.PG[t,g] - gen[g].ramp_agc, 0) for t=2:T for g=1:num_gens]
        errn = [max(-primal.PG[t-1,g] + primal.PG[t,g] - gen[g].ramp_agc, 0) for t=2:T for g=1:num_gens]
    end
    err  = [errp; errn]

    if (T <= 1 || num_gens < 1)
        return 0.0, 0.0
    end
    return norm(err, lnorm), norm(err, 1)/((T-1)*num_gens)
end

function computePrimalCost(primal::mpPrimalSolution, opfdata::OPFData; options::Option = Option())
    ## generation costs
    gencost = 0.0
    if options.obj_gencost
        gencost = computeGenerationCost(primal, opfdata; options = options)
    end

    ## penalty costs due to infeasibility
    penalty = 0.0
    if options.obj_penalty
        penalty = computePenaltyCost(primal, opfdata; options = options)
    end

    ## frequency control costs
    freqctrl = 0.0
    if options.sc_constr && options.freq_ctrl
        for t=2:length(primal.SL)
            freqctrl += 0.5*(primal.SL[t])^2
        end
    end

    totalcost = gencost + (options.weight_loadshed*penalty) + (options.weight_freqctrl*freqctrl)
    return totalcost
end

function computeGenerationCost(primal::mpPrimalSolution, opfdata::OPFData; options::Option = Option())
    gen = opfdata.generators
    baseMVA = opfdata.baseMVA
    gencost = 0.0
    for p in 1:size(primal.PG, 1)
        for g in 1:size(primal.PG, 2)
            Pg = primal.PG[p,g]
            weight = p > 1 ? options.weight_scencost : 1.0
            gencost += weight*( gen[g].coeff[gen[g].n-2]*(baseMVA*Pg)^2 +
                                gen[g].coeff[gen[g].n-1]*(baseMVA*Pg)   +
                                gen[g].coeff[gen[g].n  ])
        end
    end

    return gencost
end

function computePenaltyCost(primal::mpPrimalSolution, opfdata::OPFData; options::Option = Option())
    # TO DO
    # NEED rawdata
    #error("to do")
    return 0.0
end
