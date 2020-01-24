include("acopf.jl")

mutable struct Triplet
    i::Vector{Int}
    j::Vector{Int}
    k::Vector{Float64}
end

function runAladin(opfdata::OPFData, num_partitions::Int)
    num_partitions = max(min(length(opfdata.buses), num_partitions), 1)
    network = buildNetworkPartition(opfdata, num_partitions)
    params = initializePararms(opfdata, network)

    iter = 0
    while true
         iter += 1

        # solve NLP models
        nlpmodel = solveNLP(opfdata, network, params)

        # check convergence
        (primviol, dualviol) = computeViolation(opfdata, network, params, nlpmodel)
        if primviol <= params.tol && dualviol <= params.tol
            println("converged")
#            break
        end

        @printf("iter %d: primviol = %.2f, dualviol = %.2f\n", iter, primviol, dualviol)

        # solve QP
        solveQP(opfdata, network, params, nlpmodel)
        break
    end
end



function solveNLP(opfdata::OPFData, network::OPFNetwork, params::ALADINParams)
    nlpmodel = Vector{JuMP.Model}(undef, network.num_partitions)
    for p in 1:network.num_partitions
        nlpmodel[p] = acopf_model(opfdata, network, params, p)
        nlpmodel[p], status = acopf_solve(nlpmodel[p], opfdata, network, p)
        if status != :Optimal
            error("something went wrong with status ", status)
        end

        acopf_outputAll(nlpmodel[p], opfdata, network, p)
    end

    return nlpmodel
end



function computeViolation(opfdata::OPFData, network::OPFNetwork, params::ALADINParams, nlpmodel::Vector{JuMP.Model})
    #
    # primal violation
    #
    primviol = 0.0
    for n in network.consensus_nodes
        partition = get_prop(network.graph, n, :partition)
        blocks = get_prop(network.graph, n, :blocks)
        VMtrue = getvalue(nlpmodel[partition][:Vm])[n]
        VAtrue = getvalue(nlpmodel[partition][:Va])[n]
        for p in blocks
            (p == partition) && continue
            VMcopy = getvalue(nlpmodel[p][:Vm])[n]
            VAcopy = getvalue(nlpmodel[p][:Va])[n]
            primviol += abs(VMtrue - VMcopy) + abs(VAtrue - VAcopy)
        end
    end

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

    return (primviol, dualviol)
end

function solveQP(opfdata::OPFData, network::OPFNetwork, params::ALADINParams,
                 nlpmodel::Vector{JuMP.Model})
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

        inner = internalmodel(nlpmodel[p]).inner
        nvar_nlp = length(inner.x)
        nconstr_nlp = MathProgBase.numconstr(nlpmodel[p])

        # Evaluate the gradient of NLP p.
        grad[p] = zeros(nvar_nlp)
        MathProgBase.eval_grad_f(d[p], grad[p], inner.x)

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
        Ih, Jh = MathProgBase.hesslag_structure(d[p])
        Kh = zeros(length(Ih))
        MathProgBase.eval_hesslag(d[p], Kh, inner.x, 1.0, inner.mult_g)
        hess[p] = Triplet(Ih, Jh, Kh)
    end

    # Construct the QP model.
    qp = JuMP.Model(solver = IpoptSolver(print_level=1))

    @variable(qp, x[p=1:nNLPs, j=1:MathProgBase.numvar(nlpmodel[p])])

    # For the bound constraints, stay at the active-set.
    for p = 1:nNLPs
        m = nlpmodel[p]
        inner = internalmodel(m).inner

        for j = 1:length(inner.x)
            if abs(inner.x[j] - m.colLower[j]) <= params.zero ||
                abs(inner.x[j] - m.colUpper[j]) <= params.zero
                setlowerbound(x[p, j], 0)
                setupperbound(x[p, j], 0)
            end
        end
    end

    # Quadratic objective: 0.5*Δy_p*H_p*Δy_p for p=1..N.
    @NLexpression(qp, obj_qp_expr[p=1:nNLPs],
                  0.5*sum(hess[p].k[e]*x[p, hess[p].i[e]]*x[p, hess[p].j[e]]
                          for e=1:length(hess[p].i)))

    # Linear objective: g_p*Δy_p for p=1..N
    @NLexpression(qp, obj_g_expr[p=1:nNLPs],
                  sum(grad[p][j]*x[p, j]
                      for j=1:MathProgBase.numvar(nlpmodel[p])))

    @NLobjective(qp, Min, sum(obj_qp_expr[p] + obj_g_expr[p] for p=1:nNLPs))

    # Active constraints: C_p*Δy_p = 0 for p=1..N.
    @constraint(qp, active_constr[p=1:nNLPs, i in active_row[p]],
                sum(active_val[p][i][e]*x[p, active_col[p][i][e]]
                    for e = 1:length(active_col[p][i])) == 0)

    status = solve(qp)
    @printf("\n ## Summary of QP solve\n")
    @printf("Status  . . . . . %s\n", status)
    @printf("Objective . . . . %e\n", getobjectivevalue(qp))
end

ARGS = ["data/case9"]
opfdata = opf_loaddata(ARGS[1])
runAladin(opfdata, 1)

