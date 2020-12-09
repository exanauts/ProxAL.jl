function solve_fullmodel(opfdata::OPFData, rawdata::RawData, modelinfo::ModelParams, algparams::AlgParams)
    opfmodel = opf_model_nondecomposed(opfdata, rawdata, modelinfo, algparams)
    return opf_solve_nondecomposed(opfmodel, opfdata, modelinfo, algparams)
end

function opf_model_nondecomposed(opfdata::OPFData, rawdata::RawData, modelinfo::ModelParams, algparams::AlgParams)
    opfmodel = JuMP.Model(algparams.optimizer)
    opf_model_add_variables(opfmodel, opfdata, modelinfo, algparams)
    opf_model_add_block_constraints(opfmodel, opfdata, rawdata, modelinfo)
    obj_expr = compute_objective_function(opfmodel, opfdata, modelinfo)

    if algparams.mode == :lyapunov_bound
        lyapunov_expr = compute_quadratic_penalty(opfmodel, opfdata,  modelinfo, algparams)
        if !algparams.decompCtgs
            opf_model_add_ctgs_linking_constraints(opfmodel, opfdata, modelinfo)
        end
    else
        lyapunov_expr = 0
        opf_model_add_time_linking_constraints(opfmodel, opfdata, modelinfo)
        opf_model_add_ctgs_linking_constraints(opfmodel, opfdata, modelinfo)
    end

    @objective(opfmodel,Min, obj_expr + lyapunov_expr)

    return opfmodel
end

function opf_solve_nondecomposed(opfmodel::JuMP.Model, opfdata::OPFData,
                                 modelinfo::ModelParams,
                                 algparams::AlgParams)
    optimize!(opfmodel)
    status = termination_status(opfmodel)
    if status ∉ MOI_OPTIMAL_STATUSES
        (algparams.verbose > 0) &&
            println("warning: $(algparams.mode) model status: $status")
        return nothing
    end


    x = PrimalSolution(opfdata, modelinfo)
    x.Pg .= value.(opfmodel[:Pg])
    x.Qg .= value.(opfmodel[:Qg])
    x.Vm .= value.(opfmodel[:Vm])
    x.Va .= value.(opfmodel[:Va])
    x.ωt .= value.(opfmodel[:ωt])
    x.St .= value.(opfmodel[:St])
    x.Zt .= value.(opfmodel[:Zt])
    x.Sk .= value.(opfmodel[:Sk])
    x.Zk .= value.(opfmodel[:Zk])
    if modelinfo.allow_constr_infeas
        x.sigma_real .= value.(opfmodel[:sigma_real])
        x.sigma_imag .= value.(opfmodel[:sigma_imag])
        x.sigma_lineFr .= value.(opfmodel[:sigma_lineFr])
        x.sigma_lineTo .= value.(opfmodel[:sigma_lineTo])
    end
    # @show(maximum(abs.(x.Zt)))
    # @show(maximum(abs.(x.Zk)))
    # @show(maximum(abs.(x.ωt)))


    λ = DualSolution(opfdata, modelinfo)
    T = modelinfo.num_time_periods
    K = modelinfo.num_ctgs + 1
    if T > 1 && algparams.mode != :lyapunov_bound
        if modelinfo.time_link_constr_type == :inequality
            λ.ramping_p[:,2:T] .= collect(dual.(opfmodel[:ramping_p]))
            λ.ramping_n[:,2:T] .= collect(dual.(opfmodel[:ramping_n]))
        else
            λ.ramping[:,2:T] .= collect(dual.(opfmodel[:ramping]))
        end
    end
    if K > 1 && !(algparams.mode == :lyapunov_bound && algparams.decompCtgs)
        if modelinfo.ctgs_link_constr_type == :corrective_inequality
            λ.ctgs_p[:,2:K,:] .= collect(dual.(opfmodel[:ctgs_p]))
            λ.ctgs_n[:,2:K,:] .= collect(dual.(opfmodel[:ctgs_n]))
        else
            λ.ctgs[:,2:K,:] .= collect(dual.(opfmodel[:ctgs]))
        end
    end


    result = Dict()
    result["primal"] = x
    result["dual"] = λ
    result["objective_value_" * String(algparams.mode)] = objective_value(opfmodel)
    result["solve_time"] = JuMP.solve_time(opfmodel)

    return result
end
