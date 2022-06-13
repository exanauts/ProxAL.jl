"""
    OPFBlocks(
        opfdata::OPFData,
        rawdata::RawData;
        modelinfo::ModelInfo = ModelInfo(),
        backend=JuMPBlockBackend,
        algparams::AlgParams = AlgParams()
    )

Create a structure `OPFBlocks` to decompose the original OPF problem
specified in `opfdata` timestep by timestep, by dualizing the ramping
constraint. One block corresponds
to one optimization subproblem (and hence, to a particular timestep),
and the attribute `blkCount` enumerates the total number of subproblems.
The subproblems are specified using `AbstractBlockModel` objects,
allowing to define them either with JuMP
(if `backend=JuMPBlockBackend` is chosen) or with ExaPF (`backend=ExaBlockBackend`).


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

When the backend is set to `ExaBlockBackend` (and a CUDA GPU is available), the user
could chose to deport the resolution of each subproblem directly on
the GPU simply by setting `algparams.device=CUDADevice`. However, note that
we could not instantiate more subproblems on the GPU than the number of GPU
available.

"""
mutable struct OPFBlocks <: AbstractBlocks
    blkCount::Int64
    blkIndex::CartesianIndices
    blkModel::Vector{AbstractBlockModel}
    colCount::Int64
    colValue::Array{Float64,2}
end

function OPFBlocks(
    opfdata::OPFData,
    rawdata::RawData;
    modelinfo::ModelInfo = ModelInfo(),
    backend=JuMPBlockBackend,
    algparams::AlgParams = AlgParams(),
    comm::Union{MPI.Comm,Nothing},
)
    ngen  = length(opfdata.generators)
    nbus  = length(opfdata.buses)
    nline = length(opfdata.lines)
    T = modelinfo.num_time_periods
    K = modelinfo.num_ctgs + 1 # base case counted separately

    blkCount = algparams.decompCtgs ? (T*K) : T
    blkIndex = algparams.decompCtgs ? CartesianIndices((1:K,1:T)) : CartesianIndices((1:1,1:T))
    colCount = ((algparams.decompCtgs ? 1 : K)*(
                    2*nbus + 4*ngen + 1)) + 2*ngen
    colValue = zeros(colCount,blkCount)
    blkModel = AbstractBlockModel[]

    function local_copy(modelinfo)
        info = deepcopy(modelinfo)
        info.num_time_periods = 1
        return info
    end

    function load_local_data(
        rawdata,
        opfdata,
        modelinfo,
        t, k;
        decompCtgs=false,
    )
        lineOff = Line()
        if length(rawdata.ctgs_arr) < k - 1
            error("Not enough contingencies in .ctg file while trying to read contingency $(k-1).")
        end
        if decompCtgs
            if k > 1
                lineOff = opfdata.lines[rawdata.ctgs_arr[k - 1]]
            end
        end
        data = opf_loaddata(
            rawdata;
            time_horizon_start=modelinfo.time_horizon_start + t - 1,
            time_horizon_end=modelinfo.time_horizon_start + t - 1,
            load_scale=modelinfo.load_scale,
            ramp_scale=modelinfo.ramp_scale,
            corr_scale=modelinfo.corr_scale,
            lineOff=lineOff,
        )
        return data
    end

    for blk in LinearIndices(blkIndex)
        k = blkIndex[blk][1]
        t = blkIndex[blk][2]
        if is_my_work(blk, comm)

            # Local info
            localinfo = local_copy(modelinfo)
            if algparams.decompCtgs
                @assert k > 0
                localinfo.num_ctgs = 0
            end
            localdata = load_local_data(rawdata, opfdata, localinfo, t, k;
                                        decompCtgs=algparams.decompCtgs)
            # Create block model
            localmodel = backend(blk, localdata, rawdata, algparams, localinfo, t, k, T)
        else
            localmodel = EmptyBlockModel()
        end
        push!(blkModel, localmodel)
    end

    return OPFBlocks(blkCount, blkIndex, blkModel, colCount, colValue)
end

