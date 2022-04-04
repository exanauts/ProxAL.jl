struct NonDecomposedModel <: AbstractNLPEvaluator
    problem::ProxALProblem
    modelinfo::ModelInfo
    algparams::AlgParams
    opfdata::OPFData
    rawdata::RawData
    space::AbstractBackend
end

"""
    NonDecomposedModel(
        case_file::String,
        load_file::String,
        modelinfo::ModelInfo,
        algparams::AlgParams,
        space::AbstractBackend=JuMPBackend(),
    )

Instantiate non-decomposed multi-period ACOPF instance
specified in `case_file` with loads in `load_file` with model parameters
`modelinfo` and algorithm parameters `algparams`.

"""
function NonDecomposedModel(
    case_file::String, load_file::String,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    space::AbstractBackend=JuMPBackend(),
)
    rawdata = RawData(case_file, load_file)
    opfdata = opf_loaddata(
        rawdata;
        time_horizon_start = modelinfo.time_horizon_start,
        time_horizon_end = modelinfo.time_horizon_start + modelinfo.num_time_periods - 1,
        load_scale = modelinfo.load_scale,
        ramp_scale = modelinfo.ramp_scale,
        corr_scale = modelinfo.corr_scale
    )

    # ctgs_arr = deepcopy(rawdata.ctgs_arr)
    problem = ProxALProblem(opfdata, rawdata, modelinfo, algparams, space, nothing)
    return NonDecomposedModel(problem, modelinfo, algparams, opfdata, rawdata, space)
end

"""
    optimize!(nlp::NonDecomposedModel)

Solve problem using the `nlp` evaluator
of the nondecomposed model.

"""
function optimize!(nlp::NonDecomposedModel)

    opfmodel = opf_model_nondecomposed(nlp.opfdata, nlp.rawdata, nlp.modelinfo, nlp.algparams)
    return opf_solve_nondecomposed(opfmodel, nlp.opfdata, nlp.modelinfo, nlp.algparams)
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
    x.sigma .= value.(opfmodel[:sigma])


    result = Dict()
    result["primal"] = x
    result["objective_value_" * String(algparams.mode)] = objective_value(opfmodel)
    result["solve_time"] = JuMP.solve_time(opfmodel)

    return result
end

