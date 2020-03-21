#
# check KKT conditions manually
# NOTE: we do not check complementary slackness
#

using ForwardDiff, LinearAlgebra, SparseArrays


function generation_cost(x;
    generators,baseMVA,ngen)
    Pg = x[1:ngen]
    return 0.001*sum( generators[i].coeff[generators[i].n-2]*(baseMVA*Pg[i])^2
        +generators[i].coeff[generators[i].n-1]*(baseMVA*Pg[i])
        +generators[i].coeff[generators[i].n  ] for i in 1:ngen)
end
function grad_generation_cost(x;
    generators,baseMVA,ngen)
    g = y -> generation_cost(y;
        generators=generators,baseMVA=baseMVA,ngen=ngen)
    return ForwardDiff.gradient(g, x)
end

function real_power_balance(x;
        b,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    Pg = x[1:ngen]
    Qg = x[ngen+1:2ngen]
    Vm = x[2ngen+1:2ngen+nbus]
    Va = x[2ngen+nbus+1:2ngen+2nbus]
    return  (((isempty(FromLines[b]) ? 0 : sum( YffR[l] for l in FromLines[b])) + (isempty(ToLines[b]) ? 0 : sum( YttR[l] for l in ToLines[b])) + YshR[b] ) * Vm[b]^2) +
              (isempty(FromLines[b]) ? 0 : sum( Vm[b]*Vm[busIdx[lines[l].to]]  *( YftR[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )) +
              (isempty(ToLines[b])   ? 0 : sum( Vm[b]*Vm[busIdx[lines[l].from]]*( YtfR[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfI[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )) -
             ((isempty(BusGeners[b]) ? 0 : sum(baseMVA*Pg[g] for g in BusGeners[b])) - buses[b].Pd ) / baseMVA
end
function grad_real_power_balance(x;
        b,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    g = y -> real_power_balance(y;
            b=b,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
    return ForwardDiff.gradient(g, x)
end

function imag_power_balance(x;
        b,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    Pg = x[1:ngen]
    Qg = x[ngen+1:2ngen]
    Vm = x[2ngen+1:2ngen+nbus]
    Va = x[2ngen+nbus+1:2ngen+2nbus]
    return  (((isempty(FromLines[b]) ? 0 : sum(-YffI[l] for l in FromLines[b])) + (isempty(ToLines[b]) ? 0 : sum(-YttI[l] for l in ToLines[b])) - YshI[b] ) * Vm[b]^2) +
              (isempty(FromLines[b]) ? 0 : sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )) +
              (isempty(ToLines[b])   ? 0 : sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )) -
             ((isempty(BusGeners[b]) ? 0 : sum(baseMVA*Qg[g] for g in BusGeners[b])) - buses[b].Qd ) / baseMVA
end
function grad_imag_power_balance(x;
        b,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    g = y -> imag_power_balance(y;
            b=b,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
    return ForwardDiff.gradient(g, x)
end

function linelimit_from(x;
        l,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    Vm = x[2ngen+1:2ngen+nbus]
    Va = x[2ngen+nbus+1:2ngen+2nbus]
    flowmax=(lines[l].rateA/baseMVA)^2
    Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
    Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
    return (Vm[busIdx[lines[l].from]]^2 *
            (Yff_abs2*Vm[busIdx[lines[l].from]]^2 + Yft_abs2*Vm[busIdx[lines[l].to]]^2
                + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]]))
                )) - flowmax
end
function grad_linelimit_from(x;
        l,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    g = y -> linelimit_from(y;
            l=l,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
    return ForwardDiff.gradient(g, x)
end

function linelimit_to(x;
        l,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    Vm = x[2ngen+1:2ngen+nbus]
    Va = x[2ngen+nbus+1:2ngen+2nbus]
    flowmax=(lines[l].rateA/baseMVA)^2
    Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
    Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
    return (Vm[busIdx[lines[l].to]]^2 *
            (Ytf_abs2*Vm[busIdx[lines[l].from]]^2 + Ytt_abs2*Vm[busIdx[lines[l].to]]^2
                + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]]))
                )) - flowmax
end
function grad_linelimit_to(x;
        l,YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI,
        ngen,nbus,nline,lines,FromLines,ToLines,busIdx,baseMVA,buses,BusGeners)
    g = y -> linelimit_to(y;
            l=l,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
    return ForwardDiff.gradient(g, x)
end


function grad_Lagrangian(m::JuMP.Model, x::Vector, mult_g::Vector)
    @assert isa(internalmodel(m), Ipopt.IpoptMathProgModel)
    @assert (length(x) == MathProgBase.numvar(m))
    @assert (length(mult_g) == MathProgBase.numconstr(m))

    d = JuMP.NLPEvaluator(m)
    MathProgBase.initialize(d, [:Grad, :Jac, :Hess])
    nvar = MathProgBase.numvar(m)
    nconstr = MathProgBase.numconstr(m)

    # Evaluate the gradient
    grad = zeros(nvar)
    MathProgBase.eval_grad_f(d, grad, x)

    # Evaluate the Jacobian
    Ij_tmp, Jj_tmp = MathProgBase.jac_structure(d)
    Kj_tmp = zeros(length(Ij_tmp))
    MathProgBase.eval_jac_g(d, Kj_tmp, x)

    # Merge duplicates.
    Ij, Jj, Vj = findnz(sparse(Ij_tmp, Jj_tmp, [Int[e] for e=1:length(Ij_tmp)],
                            nconstr, nvar, vcat)
                        )
    Kj = [sum(Kj_tmp[Vj[e]]) for e=1:length(Ij)]

    # Get the KKT expression
    return grad .+ (transpose(sparse(Ij, Jj, Kj, nconstr, nvar))*mult_g)
end

function computePrimalDualError_manual(opfdata, network, nlpmodel, primal;
    lnorm = 1, compute_dual_error = true)
    Pg, Qg, Vm, Va = constructPrimalSolution(opfdata, network, primal)
    x = [Pg; Qg; Vm; Va]
    nvar = length(x)


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
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
    


    #
    # First the variable bounds
    #
    Pglb = [generators[i].Pmin for i in 1:ngen]
    Pgub = [generators[i].Pmax for i in 1:ngen]
    Qglb = [generators[i].Qmin for i in 1:ngen]
    Qgub = [generators[i].Qmax for i in 1:ngen]
    Vmlb = [buses[i].Vmin for i in 1:nbus]
    Vmub = [buses[i].Vmax for i in 1:nbus]
    Valb = -pi*ones(nbus); Valb[opfdata.bus_ref] = buses[opfdata.bus_ref].Va
    Vaub = +pi*ones(nbus); Vaub[opfdata.bus_ref] = buses[opfdata.bus_ref].Va
    lb = [Pglb; Qglb; Vmlb; Valb]
    ub = [Pgub; Qgub; Vmub; Vaub]
    bounds_error = [max(x[i] - ub[i], 0.0) + max(lb[i] - x[i], 0.0) for i in 1:nvar]


    #
    # The power balances
    #
    real_power_balance_values = zeros(nbus)
    imag_power_balance_values = zeros(nbus)
    ∇real_power_balance = zeros(nvar,nbus)
    ∇imag_power_balance = zeros(nvar,nbus)
    for b in 1:nbus
        real_power_balance_values[b] = abs(real_power_balance(x;
            b=b,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners))
        imag_power_balance_values[b] = abs(imag_power_balance(x;
            b=b,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners))

        compute_dual_error || continue

        ∇real_power_balance[:,b] = grad_real_power_balance(x;
            b=b,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
        ∇imag_power_balance[:,b] = grad_imag_power_balance(x;
            b=b,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
    end



    #
    # The line limits
    #
    linelimit_from_values = zeros(nline)
    linelimit_to_values = zeros(nline)
    ∇linelimit_from = zeros(nvar,nline)
    ∇linelimit_to = zeros(nvar,nline)
    for l in 1:nline
        if lines[l].rateA==0 || lines[l].rateA>=1.0e10
            continue
        end
        linelimit_from_values[l] = max(0.0, linelimit_from(x;
            l=l,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners))
        linelimit_to_values[l] = max(0.0, linelimit_to(x;
            l=l,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners))

        compute_dual_error || continue

        ∇linelimit_from[:,l] = grad_linelimit_from(x;
            l=l,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
        ∇linelimit_to[:,l] = grad_linelimit_to(x;
            l=l,YffR=YffR,YffI=YffI,YttR=YttR,YttI=YttI,YftR=YftR,YftI=YftI,YtfR=YtfR,YtfI=YtfI,YshR=YshR,YshI=YshI,
            ngen=ngen,nbus=nbus,nline=nline,lines=lines,FromLines=FromLines,ToLines=ToLines,busIdx=busIdx,baseMVA=baseMVA,buses=buses,BusGeners=BusGeners)
    end


    #
    # the constraint multipliers
    #
    mult_real_power_balance = zeros(nbus)
    mult_imag_power_balance = zeros(nbus)
    mult_linelimit_from = zeros(nline)
    mult_linelimit_to = zeros(nline)
    if compute_dual_error
        for p in 1:network.num_partitions
            inner = internalmodel(nlpmodel[p]).inner
            mult_g = inner.mult_g

            idx = 0
            for b in network.buses_part[p]
                idx += 1; mult_real_power_balance[b] = mult_g[idx]
                idx += 1; mult_imag_power_balance[b] = mult_g[idx]
            end

            for l in 1:nline
                if busIdx[lines[l].from] ∉ network.buses_bloc[p] ||
                    busIdx[lines[l].to] ∉ network.buses_bloc[p] ||
                    lines[l].rateA==0 ||
                    lines[l].rateA>=1.0e10
                    continue
                end
                #
                # These line limits may appear multiple times
                # in different partitions so pick a tie-breaking rule
                #
                if busIdx[lines[l].from] in network.buses_part[p]
                    idx += 1; mult_linelimit_from[l] = mult_g[idx]
                    idx += 1; mult_linelimit_to[l] = mult_g[idx]
                else
                    idx += 2
                end
            end
        end
    end


    #
    # The primal feasibility error
    #
    primal_feasibility = [bounds_error;
                          real_power_balance_values;
                          imag_power_balance_values;
                          linelimit_from_values;
                          linelimit_to_values]

    #
    # The stationarity condition error
    #
    kkt = zeros(nvar)
    if compute_dual_error
        kkt = grad_generation_cost(x;generators=generators,baseMVA=baseMVA,ngen=ngen)
        for b in 1:nbus
            kkt = kkt .+ (mult_real_power_balance[b]*∇real_power_balance[:,b])
            kkt = kkt .+ (mult_imag_power_balance[b]*∇imag_power_balance[:,b])
        end
        for l in 1:nline
            kkt = kkt .+ (mult_linelimit_from[l]*∇linelimit_from[:,l])
            kkt = kkt .+ (mult_linelimit_to[l]*∇linelimit_to[:,l])
        end
        for i in 1:nvar
            if abs(lb[i] - ub[i]) <= 1e-3
                kkt[i] = 0
            elseif abs(x[i] - lb[i]) <= 1e-3
                kkt[i] = max(0, -kkt[i])
            elseif abs(x[i] - ub[i]) <= 1e-3
                kkt[i] = max(0, kkt[i])
            else
                kkt[i] = abs(kkt[i])
            end
        end
    end


    primal_error = norm(primal_feasibility, lnorm)
    dual_error = norm(kkt, lnorm)

    if primal_error < 1e-6
        primal_error = 0.0
    end
    if dual_error < 1e-6
        dual_error = 0.0
    end

    return primal_error, dual_error
end

