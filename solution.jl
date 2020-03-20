#
# Data structures to represent primal and dual solutions
#

mutable struct DualSolution
    λVM::Dict{Tuple{Int, Int, Int}, Float64} #[(i, p, q)] := λ for the consensus constraint VM[i,p] = VM[i,q], p ≢ q
    λVA::Dict{Tuple{Int, Int, Int}, Float64} #[(i, p, q)] := λ for the consensus constraint VA[i,p] = VA[i,q], p ≢ q
end

mutable struct PrimalSolution
    PG::Vector{Dict{Int, Float64}} #[p][g] := Pg of gen g in partition p
    QG::Vector{Dict{Int, Float64}} #[p][g] := Qg of gen g in partition p
    VM::Vector{Dict{Int, Float64}} #[p][b] := Vm of bus b in partition p
    VA::Vector{Dict{Int, Float64}} #[p][b] := Va of bus b in partition p
end

function initializeDualSolution(opfdata::OPFData, network::OPFNetwork)
    λVM = Dict{Tuple{Int, Int, Int}, Float64}()
    λVA = Dict{Tuple{Int, Int, Int}, Float64}()
    for tup in network.consensus_tuple
        λVM[tup] = 0.0
        λVA[tup] = 0.0
    end

    return DualSolution(λVM, λVA)
end

function initializePrimalSolution(opfdata::OPFData, network::OPFNetwork)
    NP = network.num_partitions
    PG = Vector{Dict{Int, Float64}}(undef, NP)
    QG = Vector{Dict{Int, Float64}}(undef, NP)
    VM = Vector{Dict{Int, Float64}}(undef, NP)
    VA = Vector{Dict{Int, Float64}}(undef, NP)
    for p in 1:NP
        PG[p] = Dict{Int, Float64}()
        QG[p] = Dict{Int, Float64}()
        VM[p] = Dict{Int, Float64}()
        VA[p] = Dict{Int, Float64}()
        for g in network.gener_part[p]
            PG[p][g] = 0.5*(opfdata.generators[g].Pmax+opfdata.generators[g].Pmin)
            QG[p][g] = 0.5*(opfdata.generators[g].Qmax+opfdata.generators[g].Qmin)
        end
        for b in network.buses_bloc[p]
            VM[p][b] = 0.5*(opfdata.buses[b].Vmax+opfdata.buses[b].Vmin)
            VA[p][b] = opfdata.buses[opfdata.bus_ref].Va
        end
    end

    return PrimalSolution(PG, QG, VM, VA)
end

function perturb(primal::PrimalSolution, factor::Number)
    for p in 1:length(primal.PG)
        for g in keys(primal.PG[p])
            primal.PG[p][g] *= (1.0 + factor)
            primal.QG[p][g] *= (1.0 + factor)
        end
    end

    for p in 1:length(primal.VM)
        for b in keys(primal.VM[p])
            primal.VM[p][b] *= (1.0 + factor)
            primal.VA[p][b] *= (1.0 + factor)
        end
    end
end

function perturb(dual::DualSolution, factor::Number)
    for key in keys(dual.λVM)
        dual.λVM[key] *= (1.0 + factor)
        dual.λVA[key] *= (1.0 + factor)
    end
end

function computeDistance(x1::PrimalSolution, x2::PrimalSolution; lnorm = 1)
    xd = []
    for p in 1:length(x1.PG)
        for g in keys(x1.PG[p])
            push!(xd, x1.PG[p][g] - x2.PG[p][g])
            push!(xd, x1.QG[p][g] - x2.QG[p][g])
        end
    end
    for p in 1:length(x1.VM)
        for g in keys(x1.VM[p])
            push!(xd, x1.VM[p][g] - x2.VM[p][g])
            push!(xd, x1.VA[p][g] - x2.VA[p][g])
        end
    end

    return (isempty(xd) ? 0.0 : norm(xd, lnorm))
end

function computeDistance(x1::DualSolution, x2::DualSolution; lnorm = 1)
    xd = []
    for key in keys(x1.λVM)
        push!(xd, x1.λVM[key] - x2.λVM[key])
        push!(xd, x1.λVA[key] - x2.λVA[key])
    end

    return (isempty(xd) ? 0.0 : norm(xd, lnorm))
end

function computeDualViolation(x::PrimalSolution, xprev::PrimalSolution, nlpmodel::Vector{JuMP.Model}, network::OPFNetwork; lnorm = 1, params::AlgParams)
    dualviol = []
    for p = 1:network.num_partitions
        #
        # First get ∇_x Lagrangian
        #
        inner = internalmodel(nlpmodel[p]).inner
        kkt = grad_Lagrangian(nlpmodel[p], inner.x, inner.mult_g)

        #
        # Now adjust it so that the final quantity represents the error in the KKT conditions
        #
        if params.aladin
            # The Aug Lag part in ALADIN
            for b in network.consensus_nodes
                (b in network.buses_bloc[p]) || continue

                vm_idx = linearindex(nlpmodel[p][:Vm][b])
                va_idx = linearindex(nlpmodel[p][:Va][b])
                kkt[vm_idx] -= params.ρ*abs(x.VM[p][b] - xprev.VM[p][b])
                kkt[va_idx] -= params.ρ*abs(x.VA[p][b] - xprev.VA[p][b])
            end

        else
            # The Aug Lag part in proximal ALM
            for (b, s, t) in network.consensus_tuple
                (b in network.buses_bloc[p]) || continue
                (s!=p && t!=p) && continue

                vm_idx = linearindex(nlpmodel[p][:Vm][b])
                va_idx = linearindex(nlpmodel[p][:Va][b])

                key = (b, s, t)
                if s == p
                    if params.jacobi || t > s
                        kkt[vm_idx] -= params.ρ*(((1.0 - params.θ)*x.VM[s][b]) - (xprev.VM[t][b] - (params.θ*x.VM[t][b])))
                        kkt[va_idx] -= params.ρ*(((1.0 - params.θ)*x.VA[s][b]) - (xprev.VA[t][b] - (params.θ*x.VA[t][b])))
                    else
                        kkt[vm_idx] -= params.ρ*(1.0 - params.θ)*(x.VM[s][b] - x.VM[t][b])
                        kkt[va_idx] -= params.ρ*(1.0 - params.θ)*(x.VA[s][b] - x.VA[t][b])
                    end
                elseif t == p
                    if params.jacobi || s > t
                        kkt[vm_idx] -= params.ρ*(((1.0 - params.θ)*x.VM[t][b]) - (xprev.VM[s][b] - (params.θ*x.VM[s][b])))
                        kkt[va_idx] -= params.ρ*(((1.0 - params.θ)*x.VA[t][b]) - (xprev.VA[s][b] - (params.θ*x.VA[s][b])))
                    else
                        kkt[vm_idx] -= params.ρ*(1.0 - params.θ)*(x.VM[t][b] - x.VM[s][b])
                        kkt[va_idx] -= params.ρ*(1.0 - params.θ)*(x.VA[t][b] - x.VA[s][b])
                    end
                else
                    @assert false
                end
            end
        end
        # The proximal part in both ALADIN and proximal ALM
        for g in network.gener_part[p]
            pg_idx = linearindex(nlpmodel[p][:Pg][g])
            qg_idx = linearindex(nlpmodel[p][:Qg][g])
            kkt[pg_idx] -= params.τ*(x.PG[p][g] - xprev.PG[p][g])
            kkt[qg_idx] -= params.τ*(x.QG[p][g] - xprev.QG[p][g])
        end
        for b in network.buses_bloc[p]
            vm_idx = linearindex(nlpmodel[p][:Vm][b])
            va_idx = linearindex(nlpmodel[p][:Va][b])
            kkt[vm_idx] -= params.τ*(x.VM[p][b] - xprev.VM[p][b])
            kkt[va_idx] -= params.τ*(x.VA[p][b] - xprev.VA[p][b])
        end

        #
        # Compute the KKT error now
        #
        for j = 1:length(kkt)
            if (abs(nlpmodel[p].colLower[j] - nlpmodel[p].colUpper[j]) <= params.zero)
                kkt[j] = 0.0 # ignore
            elseif abs(inner.x[j] - nlpmodel[p].colLower[j]) <= params.zero
                kkt[j] = max(0, -kkt[j])
            elseif abs(inner.x[j] - nlpmodel[p].colUpper[j]) <= params.zero
                kkt[j] = max(0, +kkt[j])
            else
                kkt[j] = abs(kkt[j])
            end
        end
        dualviol = [dualviol; kkt]
    end

    return (isempty(dualviol) ? 0.0 : norm(dualviol, lnorm))
end

function computePrimalViolation(primal::PrimalSolution, network::OPFNetwork; lnorm = 1)

    #
    # primal violation
    #
    err = []
    for (n, p, q) in network.consensus_tuple
        push!(err, primal.VM[p][n] - primal.VM[q][n])
        push!(err, primal.VA[p][n] - primal.VA[q][n])
    end

    return (isempty(err) ? 0.0 : norm(err, lnorm))
end

function computePrimalCost(primal::PrimalSolution, opfdata::OPFData)
    generators = opfdata.generators
    baseMVA = opfdata.baseMVA
    gencost = 0.0
    for p in 1:length(primal.PG)
        for i in keys(primal.PG[p])
            Pg = primal.PG[p][i]
            gencost += generators[i].coeff[generators[i].n-2]*(baseMVA*Pg)^2 +
                       generators[i].coeff[generators[i].n-1]*(baseMVA*Pg)   +
                       generators[i].coeff[generators[i].n  ]
        end
    end

    return gencost
end

function constructPrimalSolution(opfdata::OPFData, network::OPFNetwork, primal::PrimalSolution)
    nbus  = length(opfdata.buses)
    ngen  = length(opfdata.generators)

    # construct Pg, Qg
    Pg = zeros(ngen)
    Qg = zeros(ngen)
    for g in 1:ngen
        bus_g = 0
        for b in 1:nbus
            if g in opfdata.BusGenerators[b]
                bus_g = b
                break
            end
        end
        @assert (bus_g > 0)
        p = get_prop(network.graph, bus_g, :partition)
        Pg[g] = primal.PG[p][g]
        Qg[g] = primal.QG[p][g]
    end

    # construct Vm, Va
    Vm = zeros(nbus)
    Va = zeros(nbus)
    for b in 1:nbus
        # Average the values over all partitions
        blocks = get_prop(network.graph, b, :blocks)
        for p in blocks
            Vm[b] += primal.VM[p][b]
            Va[b] += primal.VA[p][b]
        end
        Vm[b] /= length(blocks)
        Va[b] /= length(blocks)
    end

    return Pg,Qg,Vm,Va
end
