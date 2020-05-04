#
# Data structures to represent primal and dual solutions
#

mutable struct mpDualSolution
    λp::Array{Float64,2} #[t][g] := λ for the constraint (Pg[g,1] or Pg[g,t-1]) - Pg[g,t] <= r
    λn::Array{Float64,2} #[t][g] := λ for the constraint (Pg[g,1] or Pg[g,t-1]) - Pg[g,t] >= -r
end

mutable struct mpPrimalSolution
    PG::Array{Float64,2} #[t][g] := Pg of gen g in scenario/time period t
    QG::Array{Float64,2} #[t][g] := Qg of gen g in scenario/time period t
    VM::Array{Float64,2} #[t][b] := Vm of bus b in scenario/time period t
    VA::Array{Float64,2} #[t][b] := Va of bus b in scenario/time period t
    SL::Array{Float64,2} #[t][g] := Slack for (Pg[g,1] or Pg[g,t-1]) - Pg[g,t] <= r
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

    Sl = zeros(T, num_gens)
    for g=1:num_gens
        Sl[2:T,g] .= (options.sc_constr ? gen[g].scen_agc : gen[g].ramp_agc)
    end

    return mpPrimalSolution(Pg, Qg, Vm, Va, Sl)
end

function perturb(primal::mpPrimalSolution, factor::Number)
    primal.PG *= (1.0 + factor)
    primal.QG *= (1.0 + factor)
    primal.VM *= (1.0 + factor)
    primal.VA *= (1.0 + factor)
    primal.SL *= (1.0 + factor)
end

function perturb(dual::mpDualSolution, factor::Number)
    dual.λp *= (1.0 + factor)
    dual.λn *= (1.0 + factor)
    dual.λp .= max.(dual.λp, 0)
    dual.λn .= max.(dual.λn, 0)
end

function computeDistance(x1::mpPrimalSolution, x2::mpPrimalSolution; options::Option = Option(), lnorm = 1)
    if options.sc_constr
        xd = x1.PG[1,:] - x2.PG[1,:]
        return norm(xd, lnorm)
    end
    xd = [[x1.PG[t,:] - x2.PG[t,:];
           x1.QG[t,:] - x2.QG[t,:];
           x1.VM[t,:] - x2.VM[t,:];
           x1.VA[t,:] - x2.VA[t,:]] for t=1:size(x1.PG, 1)]
    return (isempty(xd) ? 0.0 : norm(vcat(xd...), lnorm))
end

function computeDistance(x1::mpDualSolution, x2::mpDualSolution; lnorm = 1)
    xd = [[x1.λp[t,:] - x2.λp[t,:];
           x1.λn[t,:] - x2.λn[t,:]] for t=1:size(x1.λ, 1)]
    return (isempty(xd) ? 0.0 : norm(vcat(xd...), lnorm))
end

function computeDualViolation(x::mpPrimalSolution, xprev::mpPrimalSolution, λ::mpDualSolution, λprev::mpDualSolution, nlpmodel::Vector{JuMP.Model}, opfdata::OPFData;
    options::Option = Option(),
    lnorm = 1,
    params::AlgParams)
    gen = opfdata.generators
    dualviol = []
    for t = 1:length(nlpmodel)
        #
        # First get ∇_x Lagrangian
        #
        inner = internalmodel(nlpmodel[t]).inner
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
        kkt = zeros(length(inner.x))

        #
        # Now adjust it so that the final quantity represents the error in the KKT conditions
        #
        if params.aladin
            error("to do")

        else
            # The Aug Lag part in proximal ALM
            for g=1:length(gen)
                idx_pg = linearindex(nlpmodel[t][:Pg][g])
                idx_sl = linearindex(nlpmodel[t][:Sl][g])
                # contingency
                if options.sc_constr
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
                            kkt[idx_pg] -= +λprev.λp[s,g]-λprev.λn[s,g] + params.ρ[t,g]*(
                                            x.PG[1,g] - xprev.PG[s,g] + xprev.SL[s,g] -
                                            gen[g].scen_agc
                                        )
                        end
                    end

                # multiperiod
                elseif options.has_ramping
                    if t > 1
                        kkt[idx_pg] += -λ.λp[t,g]+λ.λn[t,g]
                        temp = - params.ρ[t,g]*(
                                    (params.jacobi ? xprev.PG[t-1,g] : x.PG[t-1,g]) -
                                        x.PG[t,g] + x.SL[t,g] - gen[g].ramp_agc
                                    )
                        kkt[idx_pg] -= -λprev.λp[t,g]+λprev.λn[t,g]+temp
                        kkt[idx_sl] += temp
                    end
                    if t < length(nlpmodel)
                        kkt[idx_pg] += +λ.λp[t+1,g]-λ.λn[t+1,g]
                        kkt[idx_pg] -= +λprev.λ[t+1,g]-λprev.λn[t+1,g] + params.ρ[t,g]*(
                                        x.PG[t,g] - xprev.PG[t+1,g] + xprev.SL[t+1,g] -
                                        gen[g].ramp_agc
                                    )
                    end
                end
            end
        end
        # The proximal part in both ALADIN and proximal ALM
        for g=1:length(gen)
            pg_idx = linearindex(nlpmodel[t][:Pg][g])
            kkt[pg_idx] -= params.τ*(x.PG[t,g] - xprev.PG[t,g])
        end

        #
        # Compute the KKT error now
        #
        for j = 1:length(kkt)
            if (abs(nlpmodel[t].colLower[j] - nlpmodel[t].colUpper[j]) <= params.zero)
                kkt[j] = 0.0 # ignore
            elseif abs(inner.x[j] - nlpmodel[t].colLower[j]) <= params.zero
                kkt[j] = max(0, -kkt[j])
            elseif abs(inner.x[j] - nlpmodel[t].colUpper[j]) <= params.zero
                kkt[j] = max(0, +kkt[j])
            else
                kkt[j] = abs(kkt[j])
            end
        end
        dualviol = [dualviol; kkt]
    end

    if isempty(dualviol)
        return 0.0, 0.0
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
    if options.sc_constr
        errp = [max(+primal.PG[1,g] - primal.PG[t,g] - gen[g].scen_agc, 0) for t=2:T for g=1:num_gens]
        errn = [max(-primal.PG[1,g] + primal.PG[t,g] - gen[g].scen_agc, 0) for t=2:T for g=1:num_gens]
    elseif options.has_ramping
        errp = [max(+primal.PG[t-1,g] - primal.PG[t,g] - gen[g].ramp_agc, 0) for t=2:T for g=1:num_gens]
        errn = [max(-primal.PG[t-1,g] + primal.PG[t,g] - gen[g].ramp_agc, 0) for t=2:T for g=1:num_gens]
    end
    err  = [errp; errn]

    if (T <= 1 || num_gens < 1) && !isempty(err)
        return 0.0, 0.0
    end
    return norm(err, lnorm), norm(err, 1)/((T-1)*(num_gens))
end

function computePrimalCost(primal::mpPrimalSolution, opfdata::OPFData; options::Option = Option())
    gen = opfdata.generators
    baseMVA = opfdata.baseMVA
    gencost = 0.0
    arr = options.sc_constr ? [1] : (1:size(opfdata.Pd, 2))
    for p in arr
        for g in 1:size(primal.PG, 2)
            Pg = primal.PG[p,g]
            gencost += gen[g].coeff[gen[g].n-2]*(baseMVA*Pg)^2 +
                       gen[g].coeff[gen[g].n-1]*(baseMVA*Pg)   +
                       gen[g].coeff[gen[g].n  ]
        end
    end

    return gencost
end
