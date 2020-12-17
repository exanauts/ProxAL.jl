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

mutable struct OPFBlocks
    blkCount::Int64
    blkIndex::CartesianIndices
    blkModel::Vector{AbstractBlockModel}
    colCount::Int64
    colValue::Array{Float64,2}
end

function OPFBlocks(opfdata::OPFData, rawdata::RawData;
                   modelinfo::ModelParams = ModelParams(),
                   backend=JuMPBlockModel,
                   algparams::AlgParams = AlgParams())
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

