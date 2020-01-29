# Main algorithm file.

include("acopf.jl")

using SparseArrays

mutable struct Triplet
    i::Vector{Int}
    j::Vector{Int}
    k::Vector{Float64}
end

function runAladin(opfdata::OPFData, num_partitions::Int; globalization::Bool = false)
    num_partitions = max(min(length(opfdata.buses), num_partitions), 1)
    network = buildNetworkPartition(opfdata, num_partitions)
    params = initializePararms(opfdata, network)

    iter = 0
    iterlim = 1000
    verbose_level = 1

    while iter < iterlim
        iter += 1

        # solve NLP models
        nlpmodel = solveNLP(opfdata, network, params;
                             verbose_level = Int(iter%100 == 0))

        # check convergence
        primviol = computePrimalViolation(opfdata, network, params, nlpmodel)
        dualviol = computeDualViolation(opfdata, network, params, nlpmodel)
        (iter%10 == 0) && @printf("iter %d: primviol = %.2f, dualviol = %.2f\n", iter, primviol, dualviol)

        if primviol <= params.tol && dualviol <= params.tol
            println("converged")
            break
        end

        # solve QP
        qp, status = solveQP(opfdata, network, params, nlpmodel;
                             verbose_level = Int(iter%100 == 0))

        # Update primal and dual.
        if status != :Optimal
            error("QP not solved to optimality. status = ", status)
        end
        updateParams(params, opfdata, network, nlpmodel, qp; globalization = globalization)
    end
end



function solveNLP(opfdata::OPFData, network::OPFNetwork, params::ALADINParams; verbose_level = 1)
    nlpmodel = Vector{JuMP.Model}(undef, network.num_partitions)
    for p in 1:network.num_partitions
        nlpmodel[p] = acopf_model(opfdata, network, params, p)
        nlpmodel[p], status = acopf_solve(nlpmodel[p], opfdata, network, p)
        if status != :Optimal
            error("something went wrong with status ", status)
        end

        if verbose_level > 0
            acopf_outputAll(nlpmodel[p], opfdata, network, p)
        end
    end

    return nlpmodel
end


function computePrimalViolation(opfdata::OPFData, network::OPFNetwork, params::ALADINParams, nlpmodel::Vector{JuMP.Model}; evalParams::Bool = false)
    #
    # primal violation
    #
    primviol = 0.0
    for key in keys(params.λVM)
        part = get_prop(network.graph, key[1], :partition)
        if evalParams
            primviol += abs(params.VM[part][key[1]] -
                            params.VM[key[2]][key[1]])
        else
            primviol += abs(getvalue(nlpmodel[part][:Vm][key[1]]) -
                            getvalue(nlpmodel[key[2]][:Vm][key[1]]))
        end
    end
    for key in keys(params.λVA)
        part = get_prop(network.graph, key[1], :partition)
        if evalParams
            primviol += abs(params.VA[part][key[1]] -
                            params.VA[key[2]][key[1]])
        else
            primviol += abs(getvalue(nlpmodel[part][:Va][key[1]]) -
                            getvalue(nlpmodel[key[2]][:Va][key[1]]))
        end
    end

    return primviol
end

function computeDualViolation(opfdata::OPFData, network::OPFNetwork, params::ALADINParams, nlpmodel::Vector{JuMP.Model})
    #
    # dual violation
    #
    dualviol = 0.0
    for p in 1:network.num_partitions
        VM = getvalue(nlpmodel[p][:Vm])
        VA = getvalue(nlpmodel[p][:Va])
        for n in network.consensus_nodes
            (n in network.buses_bloc[p]) || continue
            dualviol += abs(VM[n] - params.VM[p][n])
            dualviol += abs(VA[n] - params.VA[p][n])
        end
    end
    dualviol *= params.ρ

    return dualviol
end

function solveQP(opfdata::OPFData, network::OPFNetwork, params::ALADINParams,
                 nlpmodel::Vector{JuMP.Model}; verbose_level = 1)
    nNLPs = length(nlpmodel)
    d = Vector{JuMP.NLPEvaluator}(undef, nNLPs)               # [p]: NLP evaluator for NLP p
    grad = Vector{Vector{Float64}}(undef, nNLPs)              # [p]: gradient value of NLP p
    gval = Vector{Vector{Float64}}(undef, nNLPs)              # [p]: constraint value of NLP p
    hess = Vector{Triplet}(undef, nNLPs)                      # [p]: vectors of triple (i, j, k)
    active_row = Vector{Vector{Int}}(undef, nNLPs)            # [p]: a list of active rows of NLP p
    active_col = Vector{Dict{Int, Vector{Int}}}(undef, nNLPs) # [p][i]: column indices of active row i of NLP p
    active_val = Vector{Dict{Int, Vector{Float64}}}(undef, nNLPs) # [p][i]: values of active row i of NLP p

    # Compute the gradient, Jacobian, and Hessian of each NLP.
    for p = 1:nNLPs
        d[p] = JuMP.NLPEvaluator(nlpmodel[p])
        MathProgBase.initialize(d[p], [:Grad, :Jac, :Hess])

        # We could generalize this to allow other solvers.
        # But, at this stage we enforce to use IPOPT.
        @assert isa(internalmodel(nlpmodel[p]), Ipopt.IpoptMathProgModel)

        inner = internalmodel(nlpmodel[p]).inner
        nvar_nlp = MathProgBase.numvar(nlpmodel[p])
        nconstr_nlp = MathProgBase.numconstr(nlpmodel[p])

        # Evaluate the gradient of NLP p.
        grad[p] = zeros(nvar_nlp)
        MathProgBase.eval_grad_f(d[p], grad[p], inner.x)

        # Remove the effect of the Lagrangian and the augmented term.
        for b in network.buses_part[p]
            (b in network.consensus_nodes) || continue
            for key in keys(params.λVM)
                (key[1] == b) || continue
                idx_vm = linearindex(nlpmodel[p][:Vm][b])
                idx_va = linearindex(nlpmodel[p][:Va][b])
                grad[p][idx_vm] -= params.λVM[key]
                grad[p][idx_vm] += params.ρ*(params.VM[p][b] - inner.x[idx_vm])
                grad[p][idx_va] -= params.λVA[key]
                grad[p][idx_va] += params.ρ*(params.VA[p][b] - inner.x[idx_va])
            end

            for j in neighbors(network.graph, b)
                (get_prop(network.graph, j, :partition) == p) && continue
                idx_vm = linearindex(nlpmodel[p][:Vm][j])
                idx_va = linearindex(nlpmodel[p][:Va][j])
                grad[p][idx_vm] += params.λVM[(j, p)]
                grad[p][idx_vm] += params.ρ*(params.VM[p][j] - inner.x[idx_vm])
                grad[p][idx_va] += params.λVA[(j, p)]
                grad[p][idx_va] += params.ρ*(params.VA[p][j] - inner.x[idx_va])
            end
        end

        # Evaluate the constraint g of NLP p.
        gval[p] = zeros(nconstr_nlp)
        MathProgBase.eval_g(d[p], gval[p], inner.x)

        # Evaluate the Jacobian of NLP p.
        Ij, Jj = MathProgBase.jac_structure(d[p])
        Kj = zeros(length(Ij))
        MathProgBase.eval_jac_g(d[p], Kj, inner.x)

        # Leave only the entries corresponding to the active-set.
        active_row[p] = [k for (k,v) in enumerate(gval[p]) if abs(v) < params.zero]
        active_col[p] = Dict{Int, Vector{Int}}()
        active_val[p] = Dict{Int, Vector{Float64}}()

        for e = 1:length(Ij)
            if Ij[e] in active_row[p]
                if haskey(active_col[p], Ij[e])
                    push!(active_col[p][Ij[e]], Jj[e])
                    push!(active_val[p][Ij[e]], Kj[e])
                else
                    active_col[p][Ij[e]] = [Jj[e]]
                    active_val[p][Ij[e]] = [Kj[e]]
                end
            end
        end

        # Evaluate the Hessian of NLP p.
        Ih_tmp, Jh_tmp = MathProgBase.hesslag_structure(d[p])
        Kh_tmp = zeros(length(Ih_tmp))
        MathProgBase.eval_hesslag(d[p], Kh_tmp, inner.x, 1.0, inner.mult_g)

        # Merge duplicates.
        Ih, Jh, Vh = findnz(sparse(Ih_tmp, Jh_tmp, [Int[e] for e=1:length(Ih_tmp)],
                                   nvar_nlp, nvar_nlp, vcat))
        Kh = zeros(length(Ih))
        for e = 1:length(Ih)
            Kh[e] = sum(Kh_tmp[Vh[e]])
        end

        #=
        # Indices of consensus nodes.
        linidx_consensus = Vector{Int}()

        for b in network.buses_bloc[p]
            (b in network.consensus_nodes) || continue
            push!(linidx_consensus, linearindex(nlpmodel[p][:Vm][b]))
            push!(linidx_consensus, linearindex(nlpmodel[p][:Va][b]))
        end
        unique!(linidx_consensus)

        # Remove the effect of the augmented term from the Hessian.
        for e = 1:length(Ih)
            if (Ih[e] == Jh[e]) && (e in linidx_consensus)
                Kh[e] -= params.ρ
            end
        end
        =#

        hess[p] = Triplet(Ih, Jh, Kh)
    end

    # Construct the QP model.
    qp = JuMP.Model(solver = IpoptSolver(print_level=1))

    # Δy_p variables for each p=1..N.
    @variable(qp, x[p=1:nNLPs, j=1:MathProgBase.numvar(nlpmodel[p])], start = 0)

    # For the bound constraints, stay values at the active-set.
    for p = 1:nNLPs
        m = nlpmodel[p]

        for j = 1:MathProgBase.numvar(m)
            if abs(m.colVal[j] - m.colLower[j]) <= params.zero ||
                abs(m.colVal[j] - m.colUpper[j]) <= params.zero
                setlowerbound(x[p, j], 0)
                setupperbound(x[p, j], 0)
            end
        end
    end

    # s variable for coupling constraints.
    part = Dict{Tuple{Int,Int}, Int}()
    for key in keys(params.λVM)
        part[key] = get_prop(network.graph, key[1], :partition)
    end

    @variable(qp, sVM[key in keys(params.λVM)], start = 0)
    @variable(qp, sVA[key in keys(params.λVA)], start = 0)

    # Quadratic objective: 0.5*Δy_p*H_p*Δy_p for p=1..N.
    @NLexpression(qp, obj_qp_expr[p=1:nNLPs],
                  0.5*sum(hess[p].k[e]*x[p, hess[p].i[e]]*x[p, hess[p].j[e]]
                          for e=1:length(hess[p].i)))

    # Linear objective: g_p*Δy_p for p=1..N
    @NLexpression(qp, obj_g_expr[p=1:nNLPs],
                  sum(grad[p][j]*x[p, j]
                      for j=1:MathProgBase.numvar(nlpmodel[p])))

    @NLexpression(qp, obj_s_expr,
                  sum(params.λVM[key]*sVM[key] for key in keys(params.λVM))
                  + (0.5*params.μ)*sum(sVM[key]^2 for key in keys(params.λVM))
                  + sum(params.λVA[key]*sVA[key] for key in keys(params.λVA))
                  + (0.5*params.μ)*sum(sVA[key]^2 for key in keys(params.λVA)))

    @NLobjective(qp, Min,
                 sum(obj_qp_expr[p] + obj_g_expr[p] for p=1:nNLPs) + obj_s_expr)

    # Coupling constraints.
    @constraint(qp, coupling_vm[key in keys(params.λVM)],
                getvalue(nlpmodel[part[key]][:Vm][key[1]])
                + x[part[key], linearindex(nlpmodel[part[key]][:Vm][key[1]])]
                - (getvalue(nlpmodel[key[2]][:Vm][key[1]])
                   + x[key[2], linearindex(nlpmodel[key[2]][:Vm][key[1]])])
                == sVM[key])

    @constraint(qp, coupling_va[key in keys(params.λVA)],
                getvalue(nlpmodel[part[key]][:Va][key[1]])
                + x[part[key], linearindex(nlpmodel[part[key]][:Va][key[1]])]
                - (getvalue(nlpmodel[key[2]][:Va][key[1]])
                   + x[key[2], linearindex(nlpmodel[key[2]][:Va][key[1]])])
                == sVA[key])

    # Active constraints: C_p*Δy_p = 0 for p=1..N.
    @constraint(qp, active_constr[p=1:nNLPs, i in active_row[p]],
                sum(active_val[p][i][e]*x[p, active_col[p][i][e]]
                    for e = 1:length(active_col[p][i])) == 0)

    status = solve(qp)

    if verbose_level > 0
        @printf("\n ## Summary of QP solve\n")
        @printf("Status  . . . . . %s\n", status)
        @printf("Objective . . . . %e\n", getobjectivevalue(qp))
    end

    return qp, status
end

function Phi(opfdata::OPFData, network::OPFNetwork, params::ALADINParams, nlpmodel::Vector{JuMP.Model}; evalParams::Bool = false)
    #
    # Compute objective
    #
    objval = 0.0
    generators = opfdata.generators
    for p in 1:network.num_partitions
        for i in network.gener_part[p]
            PG = (evalParams ? params.PG[p][i] : getvalue(nlpmodel[p][:Pg][i]))
            objval += generators[i].coeff[generators[i].n-2]*(opfdata.baseMVA*PG)^2 +
                      generators[i].coeff[generators[i].n-1]*(opfdata.baseMVA*PG) +
                      generators[i].coeff[generators[i].n  ]
        end
    end

    #
    # compute primal violation
    #
    primviol = computePrimalViolation(opfdata, network, params, nlpmodel; evalParams = evalParams)
    (primviol <= params.tol) && (primviol = 0.0)

    #
    # compute constraint violation
    #
    constrviol = 0.0
    for p in 1:length(nlpmodel)
        d = JuMP.NLPEvaluator(nlpmodel[p])
        MathProgBase.initialize(d, [:Jac])

        # re-construct x-vector
        xsol = zeros(MathProgBase.numvar(nlpmodel[p]))
        if evalParams
            for b in network.buses_bloc[p]
                xsol[linearindex(nlpmodel[p][:Vm][b])] = params.VM[p][b]
                xsol[linearindex(nlpmodel[p][:Va][b])] = params.VA[p][b]
            end
            for g in network.gener_part[p]
                xsol[linearindex(nlpmodel[p][:Pg][g])] = params.PG[p][g]
                xsol[linearindex(nlpmodel[p][:Qg][g])] = params.QG[p][g]
            end
        else
            xsol .= internalmodel(nlpmodel[p]).inner.x
        end

        # evaluate constraints
        gval = zeros(MathProgBase.numconstr(nlpmodel[p]))
        MathProgBase.eval_g(d, gval, xsol)

        # get the sign of the constraints
        (gL, gU) = JuMP.constraintbounds(nlpmodel[p])

        # evaluate constraint violation
        for i in 1:length(gval)
            @assert iszero(gL[i]) || iszero(gU[i])
            iszero(gL[i]) && (gval[i] *= -1.0) # flip sign if >= constraint
            (gval[i] < params.zero) && (gval[i] = 0)

            constrviol += max(gval[i], 0.0)
        end
    end

    return (objval, primviol, constrviol)
end

function updateParams(params::ALADINParams, opfdata::OPFData, network::OPFNetwork,
                      nlpmodel::Vector{JuMP.Model}, qp::JuMP.Model;
                      globalization::Bool = false)
    # Value of α3 from paper
    α3 = 1.0


    # Compute the old value of Phi
    if globalization
        (objval_old, primviol_old, constrviol_old) = Phi(opfdata, network, params, nlpmodel; evalParams = true)
    end

    # compute the full step
    for p in 1:length(nlpmodel)
        for b in network.buses_bloc[p]
            params.VM[p][b] = getvalue(nlpmodel[p][:Vm][b]) +
                getvalue(qp[:x][p, linearindex(nlpmodel[p][:Vm][b])])
            params.VA[p][b] = getvalue(nlpmodel[p][:Va][b]) +
                getvalue(qp[:x][p, linearindex(nlpmodel[p][:Va][b])])
        end
        for g in network.gener_part[p]
            params.PG[p][g] = getvalue(nlpmodel[p][:Pg][g]) +
                getvalue(qp[:x][p, linearindex(nlpmodel[p][:Pg][g])])
            params.QG[p][g] = getvalue(nlpmodel[p][:Qg][g]) +
                getvalue(qp[:x][p, linearindex(nlpmodel[p][:Qg][g])])
        end
    end

    # Compute the new value of Phi at the full step
    if globalization
        (objval_new, primviol_new, constrviol_new) = Phi(opfdata, network, params, nlpmodel; evalParams = true)
    end


    # Check if full step should be accepted
    if (globalization && 
            objval_new >= objval_old && 
            primviol_new >= primviol_old && 
            constrviol_new >= constrviol_old)
        # reject full step
        (objval_nlp, primviol_nlp, constrviol_nlp) = Phi(opfdata, network, params, nlpmodel; evalParams = false)

        # Check if NLP step can be accepted
        if objval_nlp >= objval_old && primviol_nlp >= primviol_old && constrviol_nlp >= constrviol_old
            # reject NLP step also -- default to full step
            #println("both steps rejected -- descent not guaranteed")
            α3 = 1.0
        else
            # accept NLP step
            α3 = 0.0 # should this be 1???
            #println("nlp step accepted -- Improvement = ", objval_old - objval_nlp, " ", primviol_old - primviol_nlp, " ", constrviol_old - constrviol_nlp)
            for p in 1:length(nlpmodel)
                for b in network.buses_bloc[p]
                    params.VM[p][b] = getvalue(nlpmodel[p][:Vm][b])
                    params.VA[p][b] = getvalue(nlpmodel[p][:Va][b])
                end
                for g in network.gener_part[p]
                    params.PG[p][g] = getvalue(nlpmodel[p][:Pg][g])
                    params.QG[p][g] = getvalue(nlpmodel[p][:Qg][g])
                end
            end
        end
    end

    # Update the Lagrange multipliers
    in_qp = internalmodel(qp).inner
    for key in keys(params.λVM)
        params.λVM[key] += α3*(in_qp.mult_g[linearindex(qp[:coupling_vm][key])] - params.λVM[key])
        params.λVA[key] += α3*(in_qp.mult_g[linearindex(qp[:coupling_va][key])] - params.λVA[key])
    end
end


ARGS = ["data/case9"]
opfdata = opf_loaddata(ARGS[1])
runAladin(opfdata, 3; globalization = false)

