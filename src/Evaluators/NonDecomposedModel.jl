struct NonDecomposedModel <: AbstractNLPEvaluator
    problem::ProxALProblem
    modelinfo::ModelInfo
    algparams::AlgParams
    opfdata::OPFData
    rawdata::RawData
end

"""
    NonDecomposedModel(
        case_file::String,
        load_file::String,
        modelinfo::ModelInfo,
        algparams::AlgParams,
    )

Instantiate non-decomposed multi-period ACOPF instance
specified in `case_file` with loads in `load_file` with model parameters
`modelinfo` and algorithm parameters `algparams`.
The model is solved with the `JuMPBackend`,
see [NLP blocks and backends](@ref).

"""
function NonDecomposedModel(
    case_file::String, load_file::String,
    modelinfo::ModelInfo,
    algparams::AlgParams,
)
    rawdata = RawData(case_file, load_file)
    return NonDecomposedModel(rawdata, modelinfo, algparams)
end

"""
    NonDecomposedModel(
        rawdata::RawData
        modelinfo::ModelInfo,
        algparams::AlgParams,
    )

Instantiate non-decomposed multi-period ACOPF instance
using `rawdata` with model parameters
`modelinfo` and algorithm parameters `algparams`.
The model is solved with the `JuMPBackend`,
see [NLP blocks and backends](@ref).

"""
function NonDecomposedModel(
    rawdata::RawData,
    modelinfo::ModelInfo,
    algparams::AlgParams,
)
    opfdata = opf_loaddata(
        rawdata;
        time_horizon_start = modelinfo.time_horizon_start,
        time_horizon_end = modelinfo.time_horizon_start + modelinfo.num_time_periods - 1,
        load_scale = modelinfo.load_scale,
        ramp_scale = modelinfo.ramp_scale,
        corr_scale = modelinfo.corr_scale
    )

    problem = ProxALProblem(opfdata, rawdata, modelinfo, algparams, JuMPBackend(), nothing)
    return NonDecomposedModel(problem, modelinfo, algparams, opfdata, rawdata)
end

"""
    optimize!(nlp::NonDecomposedModel)

Solve problem using the `nlp` evaluator
of the nondecomposed model.

"""
function optimize!(nlp::NonDecomposedModel)

    opfmodel = opf_model_nondecomposed(nlp.opfdata, nlp.rawdata, nlp.modelinfo, nlp.algparams)
    result = opf_solve_nondecomposed(opfmodel, nlp.opfdata, nlp.modelinfo, nlp.algparams)
    nlp.problem.x = result["primal"]
    nlp.problem.wall_time_elapsed_actual = result["solve_time"]
    push!(nlp.problem.objvalue, objective_value(opfmodel))
    return nlp.problem
end

function opf_model_nondecomposed(opfdata::OPFData, rawdata::RawData, modelinfo::ModelInfo, algparams::AlgParams)
    opfmodel = JuMP.Model(algparams.optimizer)
    opf_model_add_variables(opfmodel, opfdata, modelinfo, algparams)
    opf_model_add_block_constraints(opfmodel, opfdata, rawdata, modelinfo)
    obj_expr = compute_objective_function(opfmodel, opfdata, modelinfo, algparams)

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

    @objective(opfmodel, Min, obj_expr + lyapunov_expr)

    return opfmodel
end

function opf_solve_nondecomposed(opfmodel::JuMP.Model, opfdata::OPFData,
                                 modelinfo::ModelInfo,
                                 algparams::AlgParams)
    JuMP.optimize!(opfmodel)
    status = termination_status(opfmodel)
    if status ∉ MOI_OPTIMAL_STATUSES
        (algparams.verbose > 0) &&
            @warn("$(algparams.mode) model not solved to optimality. status: $status")
        return nothing
    end


    x = OPFPrimalSolution(opfdata, modelinfo)
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


    result = Dict()
    result["primal"] = x
    result["objective_value_" * String(algparams.mode)] = objective_value(opfmodel)
    result["solve_time"] = JuMP.solve_time(opfmodel)

    return result
end

