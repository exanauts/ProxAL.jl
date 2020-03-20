#
# ALADIN implementation
#

using SparseArrays

mutable struct Triplet
    i::Vector{Int}
    j::Vector{Int}
    k::Vector{Float64}
end

function runAladin(case::String, num_partitions::Int, perturbation::Number = 0.1)
    opfdata = opf_loaddata(case)
    num_partitions = max(min(length(opfdata.buses), num_partitions), 1)
    network = buildNetworkPartition(opfdata, num_partitions)

    #
    # start from perturbation of optimal solution
    #
    monolithic = acopf_model_monolithic(opfdata, network)
    xstar, λstar = acopf_solve_monolithic(monolithic, opfdata, network)
    @printf("Optimal generation cost = %.2f\n", computePrimalCost(xstar, opfdata))
    
    x = deepcopy(xstar); perturb(x, perturbation)
    λ = deepcopy(λstar); perturb(λ, perturbation)

    #
    # Start from scratch
    #
    #x = initializePrimalSolution(opfdata, network)
    #λ = initializeDualSolution(opfdata, network)


    verbose_level = 0
    
    plt = nothing
    if verbose_level > 1
        plt = initializePlot_iterative()
    end



    #
    # Initialize algorithmic parameters
    #
    if num_partitions > 1
        ρVM = maximum([abs(λstar.λVM[key]) for key in keys(λstar.λVM)])
        ρVA = maximum([abs(λstar.λVA[key]) for key in keys(λstar.λVA)])
        stepρ = max(ρVM, ρVA) 
        maxρ = 25.0stepρ
        #maxρ = 5.0num_partitions #case30
    else
        stepρ = 0.0
        maxρ = 0.0
    end
    params = initializeParams(maxρ; aladin = true, jacobi = true)
    savedata = zeros(params.iterlim, 6)
    xnlp = initializePrimalSolution(opfdata, network)
    xqp = initializePrimalSolution(opfdata, network)
    tstart = time()
    timeNLP = 0.0
    timeQP = 0.0
    iter = 0
    while iter < params.iterlim
        iter += 1


        #
        # solve NLP models
        #
        nlpmodel = Vector{JuMP.Model}(undef, num_partitions)
        for p in 1:num_partitions
            nlpmodel[p] = acopf_model(opfdata, network, p; params = params, primal = x, dual = λ)
            t0 = time()
            nlpmodel[p], status = acopf_solve(nlpmodel[p], opfdata, network, p; initial = x)
            t1 = time(); timeNLP += t1 - t0
            if status != :Optimal && status != :UserLimit
                error("something went wrong in the x-update of ALADIN with status ", status)
            end
        end
        #

        #
        # get NLP solution
        #
        xnlp = deepcopy(x)
        for p in 1:num_partitions
            updatePrimalSolution(xnlp, nlpmodel, network, p)
        end


        #
        # check convergence
        #
        dist = computeDistance(xnlp, xstar; lnorm=Inf)
        primviol = computePrimalViolation(xnlp, network; lnorm = Inf)
        dualviol = computeDualViolation(xnlp, x, nlpmodel, network; lnorm = Inf, params = params)
        if verbose_level > 1
            updatePlot_iterative(plt, iter, dist, primviol, dualviol)#, xavg_primviol, xavg_dualviol)
        end

        #
        # Various termination criteria
        converged = min(dist, max(primviol, dualviol)) <= params.tol
        if verbose_level > 0 || converged
            @printf("iter %d: primviol = %.2f, dualviol = %.2f, gencost = %.2f, dist = %.3f\n", iter, primviol, dualviol, computePrimalCost(xnlp, opfdata), dist)
        end
        if false
            x = deepcopy(xnlp)
            @printf("converged\n")
            break
        end  




        #
        # Solve QP
        #
        t2 = time()
        qp, status = solveQP(xnlp, nlpmodel, network, x, λ;
                                verbose_level = 0, params = params,
                                enforce_equality = false, enforce_convexity = false)
        #
        # If unbounded, attempt to enforce
        # the consensus constraints as strict equalities
        #
        if status == :Unbounded
            qp, status = solveQP(xnlp, nlpmodel, network, x, λ;
                                    verbose_level = 0, params = params,
                                    enforce_equality = true, enforce_convexity = false)
        end
        #
        # If infeasible, try again by making
        # the Hessian convex by adding diagonal terms
        #
        if status == :Infeasible || status == :UserLimit
            while true
                params.μ *= 2.0
                params.θ /= 2.0
                @printf("increasing μ = %.2f\n", params.μ)
                qp, status = solveQP(xnlp, nlpmodel, network, x, λ;
                                    verbose_level = 0, params = params,
                                    enforce_equality = false, enforce_convexity = true)
                if status == :Unbounded || status == :UserLimit
                    continue
                else
                    break
                end
            end
        end
        t3 = time(); timeQP += t3 - t2
        if status != :Optimal
            error("QP not solved to optimality. status = ", status)
        end




        #
        # Compute the qp solution
        #
        for p in 1:num_partitions
            for b in network.buses_bloc[p]
                xqp.VM[p][b] = getvalue(qp[:x][p, linearindex(nlpmodel[p][:Vm][b])])
                xqp.VA[p][b] = getvalue(qp[:x][p, linearindex(nlpmodel[p][:Vm][b])])
            end
            for g in network.gener_part[p]
                xqp.PG[p][g] = getvalue(qp[:x][p, linearindex(nlpmodel[p][:Pg][g])])
                xqp.QG[p][g] = getvalue(qp[:x][p, linearindex(nlpmodel[p][:Qg][g])])
            end
        end




        #
        # Update primal and dual
        #
        updateAladinPrimalSolution(x, xnlp, xqp)
        updateAladinDualSolution(x, λ; params = params)

        


        #
        # Update parameters (optional)
        #
        if params.updateρ && primviol > 0.1
            if params.ρ + stepρ <= maxρ
                params.ρ += stepρ
                params.τ += stepρ
                params.μ += stepρ
            end
        end

        savedata[iter,:] = [dist, primviol, dualviol, timeNLP, timeQP, time() - tstart]
    end

    writedlm(getDataFilename("", case, "aladin", num_partitions, perturbation, params.jacobi), savedata)

    return x, λ
end

function solveQP(xnlp::PrimalSolution, nlpmodel::Vector{JuMP.Model}, network::OPFNetwork, xprev::PrimalSolution, dual::DualSolution;
                    verbose_level = 1, params::AlgParams, enforce_equality=false, enforce_convexity=false)
    NP = network.num_partitions
    grad = Vector{Vector{Float64}}(undef, NP)              # [p]: gradient value of NLP p
    gval = Vector{Vector{Float64}}(undef, NP)              # [p]: constraint value of NLP p
    hess = Vector{Triplet}(undef, NP)                      # [p]: vectors of triple (i, j, k)
    active_row = Vector{Vector{Int}}(undef, NP)            # [p]: a list of active rows of NLP p
    active_col = Vector{Dict{Int, Vector{Int}}}(undef, NP) # [p][i]: column indices of active row i of NLP p
    active_val = Vector{Dict{Int, Vector{Float64}}}(undef, NP) # [p][i]: values of active row i of NLP p

    # Compute the gradient, Jacobian, and Hessian of each NLP.
    for p = 1:NP
        d = JuMP.NLPEvaluator(nlpmodel[p])
        MathProgBase.initialize(d, [:Grad, :Jac, :Hess])

        # We could generalize this to allow other solvers.
        # But, at this stage we enforce to use IPOPT.
        @assert isa(internalmodel(nlpmodel[p]), Ipopt.IpoptMathProgModel)

        inner = internalmodel(nlpmodel[p]).inner
        nvar_nlp = MathProgBase.numvar(nlpmodel[p])
        nconstr_nlp = MathProgBase.numconstr(nlpmodel[p])

        # Evaluate the gradient of NLP p.
        grad[p] = zeros(nvar_nlp)
        MathProgBase.eval_grad_f(d, grad[p], inner.x)

        #
        # Remove the effect of the Lagrangian and the augmented term.
        #
        for (b, s, t) in network.consensus_tuple
            (b in network.buses_bloc[p]) || continue
            (s!=p && t!=p) && continue

            idx_vm = linearindex(nlpmodel[p][:Vm][b])
            idx_va = linearindex(nlpmodel[p][:Va][b])

            key = (b, s, t)
            coef = (s == p) ? 1.0 : -1.0
            grad[p][idx_vm] -= coef*dual.λVM[key] + (params.ρ*(xnlp.VM[p][b] - xprev.VM[p][b]))
            grad[p][idx_va] -= coef*dual.λVA[key] + (params.ρ*(xnlp.VA[p][b] - xprev.VA[p][b]))
        end
        for g in network.gener_part[p]
            idx_pg = linearindex(nlpmodel[p][:Pg][g])
            idx_qg = linearindex(nlpmodel[p][:Qg][g])

            grad[p][idx_pg] -= params.τ*(xnlp.PG[p][g] - xprev.PG[p][g])
            grad[p][idx_qg] -= params.τ*(xnlp.QG[p][g] - xprev.QG[p][g])
        end
        for b in network.buses_bloc[p]
            idx_vm = linearindex(nlpmodel[p][:Vm][b])
            idx_va = linearindex(nlpmodel[p][:Va][b])

            grad[p][idx_vm] -= params.τ*(xnlp.VM[p][b] - xprev.VM[p][b])
            grad[p][idx_va] -= params.τ*(xnlp.VA[p][b] - xprev.VA[p][b])
        end



        # Evaluate the constraint g of NLP p.
        gval = zeros(nconstr_nlp)
        MathProgBase.eval_g(d, gval, inner.x)


        # Evaluate the Jacobian of NLP p.
        Ij, Jj = MathProgBase.jac_structure(d)
        Kj = zeros(length(Ij))
        MathProgBase.eval_jac_g(d, Kj, inner.x)


        # Leave only the entries corresponding to the active-set.
        active_row[p] = [k for (k,v) in enumerate(gval) if abs(v) < params.zero]
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
        Ih_tmp, Jh_tmp = MathProgBase.hesslag_structure(d)
        Kh_tmp = zeros(length(Ih_tmp))
        MathProgBase.eval_hesslag(d, Kh_tmp, inner.x, 1.0, inner.mult_g)

        # Merge duplicates.
        Ih, Jh, Vh = findnz(sparse(Ih_tmp, Jh_tmp, [Int[e] for e=1:length(Ih_tmp)],
                                   nvar_nlp, nvar_nlp, vcat))
        Kh = zeros(length(Ih))
        for e = 1:length(Ih)
            Kh[e] = sum(Kh_tmp[Vh[e]])
        end

        # Indices of consensus nodes.
        linidx_consensus = Vector{Int}()
        for b in network.buses_bloc[p]
            (b in network.consensus_nodes) || continue
            push!(linidx_consensus, linearindex(nlpmodel[p][:Vm][b]))
            push!(linidx_consensus, linearindex(nlpmodel[p][:Va][b]))
        end
        unique!(linidx_consensus)

        # Remove the effect of the augmented term from the Hessian.
        # ---->
        # Let's not remove the effect of ρ so as to
        # keep the QP convex along the critical cone
        # (hopefully IPOPT ensures that Hess L is PSD in the critical cone)
        #=
        for e = 1:length(Ih)
            if (Ih[e] == Jh[e])
                Kh[e] -= params.τ
                if e in linidx_consensus
                    Kh[e] -= params.ρ
                end
            end
        end
        =#


        #
        # Make the Hessian convex
        #
        if enforce_convexity
            for e = 1:length(Ih)
                if (Ih[e] == Jh[e])
                    Kh[e] += params.μ
                end
            end
        end

        hess[p] = Triplet(Ih, Jh, Kh)
    end

    # Construct the QP model.
    qp = JuMP.Model(solver = IpoptSolver(print_level=1))

    # Δy_p variables for each p=1..N.
    @variable(qp, x[p=1:NP, j=1:MathProgBase.numvar(nlpmodel[p])], start = 0)

    # For the bound constraints, stay values at the active-set.
    for p = 1:NP
        m = nlpmodel[p]

        for j = 1:MathProgBase.numvar(m)
            if abs(m.colVal[j] - m.colLower[j]) <= params.zero ||
                abs(m.colVal[j] - m.colUpper[j]) <= params.zero
                setlowerbound(x[p, j], 0)
                setupperbound(x[p, j], 0)
            end
        end
    end



    # OPTION 1
    # directly incorporate coupling constraints
    # in the objective function
    if true
        @NLexpression(qp, sVM[key in keys(dual.λVM)],
                    +(xnlp.VM[key[2]][key[1]] + x[key[2], linearindex(nlpmodel[key[2]][:Vm][key[1]])])
                    -(xnlp.VM[key[3]][key[1]] + x[key[3], linearindex(nlpmodel[key[3]][:Vm][key[1]])])
                    )
        @NLexpression(qp, sVA[key in keys(dual.λVA)], 
                    +(xnlp.VA[key[2]][key[1]] + x[key[2], linearindex(nlpmodel[key[2]][:Va][key[1]])])
                    -(xnlp.VA[key[3]][key[1]] + x[key[3], linearindex(nlpmodel[key[3]][:Va][key[1]])])
                    )
    # OPTION 2
    # Add explicit s variables for
    # coupling constraints
    else
        @variable(qp, sVM[key in keys(dual.λVM)], start=0)
        @variable(qp, sVA[key in keys(dual.λVM)], start=0)
        @constraint(qp, coupling_vm[key in keys(dual.λVM)],
                    +(xnlp.VM[key[2]][key[1]] + x[key[2], linearindex(nlpmodel[key[2]][:Vm][key[1]])])
                    -(xnlp.VM[key[3]][key[1]] + x[key[3], linearindex(nlpmodel[key[3]][:Vm][key[1]])])
                    == sVM[key])

        @constraint(qp, coupling_va[key in keys(dual.λVA)],
                    +(xnlp.VA[key[2]][key[1]] + x[key[2], linearindex(nlpmodel[key[2]][:Va][key[1]])])
                    -(xnlp.VA[key[3]][key[1]] + x[key[3], linearindex(nlpmodel[key[3]][:Va][key[1]])])
                    == sVA[key])
    end



    # Quadratic objective: 0.5*Δy_p*H_p*Δy_p for p=1..N.
    @NLexpression(qp, obj_qp_expr[p=1:NP],
                  0.5*sum(hess[p].k[e]*x[p, hess[p].i[e]]*x[p, hess[p].j[e]]
                          for e=1:length(hess[p].i)))

    # Linear objective: g_p*Δy_p for p=1..N
    @NLexpression(qp, obj_g_expr[p=1:NP],
                  sum(grad[p][j]*x[p, j]
                      for j=1:MathProgBase.numvar(nlpmodel[p])))


    if enforce_equality
        @NLobjective(qp, Min,
                     sum(obj_qp_expr[p] + obj_g_expr[p] for p=1:NP))
        @NLconstraint(qp, [key in keys(dual.λVM)], sVM[key] == 0)
        @NLconstraint(qp, [key in keys(dual.λVA)], sVA[key] == 0)
    else
        # coupling expression in objective
        @NLexpression(qp, obj_s_expr,
                      sum(dual.λVM[key]*sVM[key] for key in keys(dual.λVM))
                      + (0.5*params.μ)*sum(sVM[key]^2 for key in keys(dual.λVM))
                      + sum(dual.λVA[key]*sVA[key] for key in keys(dual.λVA))
                      + (0.5*params.μ)*sum(sVA[key]^2 for key in keys(dual.λVA)))
        @NLobjective(qp, Min,
                     sum(obj_qp_expr[p] + obj_g_expr[p] for p=1:NP) + obj_s_expr)
    end


    # Active constraints: C_p*Δy_p = 0 for p=1..N.
    @constraint(qp, active_constr[p=1:NP, i in active_row[p]],
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

function updateAladinPrimalSolution(x::PrimalSolution, xnlp::PrimalSolution, xqp::PrimalSolution)
    # primal update
    for p in 1:length(x.VM)
        for b in keys(x.VM[p])
            x.VM[p][b] = xnlp.VM[p][b] + xqp.VM[p][b]
            x.VA[p][b] = xnlp.VA[p][b] + xqp.VA[p][b]
        end
        for g in keys(x.PG[p])
            x.PG[p][g] = xnlp.PG[p][g] + xqp.PG[p][g]
            x.QG[p][g] = xnlp.QG[p][g] + xqp.QG[p][g]
        end
    end
end

function updateAladinDualSolution(x::PrimalSolution, dual::DualSolution; params::AlgParams)
    # dual update
    for (b, p, q) in keys(dual.λVM)
        key = (b, p, q)
        dual.λVM[key] += params.θ*params.μ*(x.VM[p][b] - x.VM[q][b])
        dual.λVA[key] += params.θ*params.μ*(x.VA[p][b] - x.VA[q][b])
    end
end

#=
function Phi(opfdata::OPFData, network::OPFNetwork, params::AlgParams, nlpmodel::Vector{JuMP.Model}; evalParams::Bool = false)
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

function updateParams(params::AlgParams, opfdata::OPFData, network::OPFNetwork, nlpmodel::Vector{JuMP.Model}, qp::JuMP.Model; globalization::Bool = false)
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
=#

