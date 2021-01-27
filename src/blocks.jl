function localcopy(modelinfo)
    info = deepcopy(modelinfo)
    info.num_time_periods = 1
    return info
end

function load_local_data(rawdata, opfdata, modelinfo, t, k; decompCtgs=false)
    lineOff = Line()
    if decompCtgs
        if k > 1
            lineOff = opfdata.lines[rawdata.ctgs_arr[k - 1]]
        end
    end
    data = opf_loaddata(
        rawdata;
        time_horizon_start=t,
        time_horizon_end=t,
        load_scale=modelinfo.load_scale,
        ramp_scale=modelinfo.ramp_scale,
        lineOff=lineOff,
    )
    return data
end

"""
    OPFBlocks(
        opfdata::OPFData,
        rawdata::RawData;
        modelinfo::ModelParams = ModelParams(),
        backend=JuMPBlockModel,
        algparams::AlgParams = AlgParams()
    )

Create a structure `OPFBlocks` to decompose the original OPF problem
specified in `opfdata` timestep by timestep, by dualizing the ramping
constraint. One block corresponds
to one optimization subproblem (and hence, to a particular timestep),
and the attribute `blkCount` enumerates the total number of subproblems.
The subproblems are specified using `AbstractBlockModel` objects,
allowing to define them either with JuMP
(if `backend=JuMPBlockModel` is chosen) or with ExaPF (`backend=ExaBlockModel`).


### Decomposition by contingencies

By default, OPFBlocks decomposes the problem only timestep by
timestep (single-period multiple-contingency scheme),
leading to a total of `T` subproblems.
However, if the option `algparams.decompCtgs` is set to `true`,
the original problem is also decomposed contingency by contingency
(single-period single-contingency scheme).
In this case the total number of subproblems is `T * K` (with `K` the
total number of contingencies).


### Deporting the resolution on the GPU

When the backend is set to `ExaBlockModel` (and a CUDA GPU is available), the user
could chose to deport the resolution of each subproblem directly on
the GPU simply by setting `algparams.device=CUDADevice`. However, note that
we could not instantiate more subproblems on the GPU than the number of GPU
available.

"""
mutable struct OPFBlocks
    blkCount::Int64
    blkIndex::CartesianIndices
    blkModel::Vector{AbstractBlockModel}
    colCount::Int64
    colValue::Array{Float64,2}
end

function OPFBlocks(
    opfdata::OPFData,
    rawdata::RawData;
    modelinfo::ModelParams = ModelParams(),
    backend=JuMPBlockModel,
    algparams::AlgParams = AlgParams()
)
    ngen  = length(opfdata.generators)
    nbus  = length(opfdata.buses)
    nline = length(opfdata.lines)
    T = modelinfo.num_time_periods
    K = modelinfo.num_ctgs + 1 # base case counted separately

    blkCount = algparams.decompCtgs ? (T*K) : T
    blkIndex = algparams.decompCtgs ? CartesianIndices((1:K,1:T)) : CartesianIndices((1:1,1:T))
    colCount = ((algparams.decompCtgs ? 1 : K)*(
                    2nbus + 4ngen + 1 +
                        (Int(modelinfo.allow_constr_infeas)*(2nbus + 2nline))
                )) + 2ngen
    colValue = zeros(colCount,blkCount)
    blkModel = AbstractBlockModel[]

    for blk in LinearIndices(blkIndex)
        k = blkIndex[blk][1]
        t = blkIndex[blk][2]

        # Local info
        localinfo = localcopy(modelinfo)
        if algparams.decompCtgs
            @assert k > 0
            localinfo.num_ctgs = 0
        end
        localdata = load_local_data(rawdata, opfdata, localinfo, t, k;
                                    decompCtgs=algparams.decompCtgs)
        # Create block model
        localmodel = backend(blk, localdata, rawdata, localinfo, t, k, T;
                             device=algparams.device, nr_tol=algparams.nr_tol)
        push!(blkModel, localmodel)
    end

    return OPFBlocks(blkCount, blkIndex, blkModel, colCount, colValue)
end

