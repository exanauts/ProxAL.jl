using DelimitedFiles

struct Bus
    bus_i::Int
    bustype::Int
    Pd::Float64
    Qd::Float64
    Gs::Float64
    Bs::Float64
    area::Int
    Vm::Float64
    Va::Float64
    baseKV::Float64
    zone::Int
    Vmax::Float64
    Vmin::Float64  
end

struct Line
    from::Int
    to::Int
    r::Float64
    x::Float64
    b::Float64
    rateA::Float64
    rateB::Float64
    rateC::Float64
    ratio::Float64 #TAP
    angle::Float64 #SHIFT
    status::Int
    angmin::Float64
    angmax::Float64
end
Line() = Line(0,0,0.,0.,0.,0.,0.,0.,0.,0.,0,0.,0.)

mutable struct Gener
    # .gen fields
    bus::Int
    Pg::Float64
    Qg::Float64
    Qmax::Float64
    Qmin::Float64
    Vg::Float64
    mBase::Float64
    status::Int
    Pmax::Float64
    Pmin::Float64
    Pc1::Float64
    Pc2::Float64
    Qc1min::Float64
    Qc1max::Float64
    Qc2min::Float64
    Qc2max::Float64
    ramp_agc::Float64
    scen_agc::Float64 #ramping factor under contingency
    # .gencost fields
    gentype::Int
    startup::Float64
    shutdown::Float64
    n::Int 
    coeff::Array
    alpha::Float64
end

struct OPFData
    buses::Array{Bus}
    lines::Array{Line}
    generators::Array{Gener}
    bus_ref::Int
    baseMVA::Float64
    BusIdx::Dict{Int,Int}    #map from bus ID to bus index
    FromLines::Array         #From lines for each bus (Array of Array)
    ToLines::Array           #To lines for each bus (Array of Array)
    BusGenerators::Array     #list of generators for each bus (Array of Array)
    Pd::Array                #2d array of active power demands over a time horizon
    Qd::Array                #2d array of reactive power demands
end


mutable struct Option
    obj_gencost::Bool
    obj_penalty::Bool
    has_ramping::Bool
    phase1::Bool
    freq_ctrl::Bool
    load_shed::Bool
    sc_constr::Bool
    piecewise::Bool
    two_block::Bool
    powerflow_solve::Bool
    weight_scencost::Float64
    weight_loadshed::Float64
    weight_freqctrl::Float64
    savefile::String
    prev_val::Vector{Float64}
    neg_g::Vector{Int}

    function Option()
        new(true,  # obj_gencost
            false, # obj_penalty
            false, # has_ramping
            false, # phase1
            false, # freq_ctrl
            false, # load_shed
            false, # sc_constr
            false, # piecewise
            false, # two_block
            false, # powerflow_solve
            1.0,   # weight_scencost
            1.0,   # weight_loadshed
            1.0,   # weight_freqctrl
            "",    # savefile
            Vector{Float64}(), # prev_val
            Vector{Int}()      # neg_g
        )
    end
end


mutable struct RawData
    bus_arr
    branch_arr
    gen_arr
    costgen_arr
    pd_arr
    qd_arr
    ctgs_arr
end

function RawData(case_name, scen_file::String="")
    bus_arr = readdlm(case_name * ".bus")
    branch_arr = readdlm(case_name * ".branch")
    gen_arr = readdlm(case_name * ".gen")
    costgen_arr = readdlm(case_name * ".gencost")
    pd_arr = Array{Float64, 2}(undef, 0, 0)
    qd_arr = Array{Float64, 2}(undef, 0, 0)
    ctgs_arr = Array{Int64, 1}(undef, 0)
    if isfile(scen_file * ".Pd")
        pd_arr = readdlm(scen_file * ".Pd")
    end
    if isfile(scen_file * ".Qd")
        qd_arr = readdlm(scen_file * ".Qd")
    end
    if isfile(scen_file * ".Ctgs")
        ctgs_arr = readdlm(scen_file * ".Ctgs", Int)
    end
    return RawData(bus_arr, branch_arr, gen_arr, costgen_arr, pd_arr, qd_arr, ctgs_arr)
end

function ctgs_loaddata(raw::RawData, n)
    return raw.ctgs_arr[1:n]
end

function opf_loaddata(raw::RawData; time_horizon::Int=0, load_scale::Float64=1.0, ramp_scale::Float64=0.0, lineOff=Line())
    #
    # load buses
    #
    bus_arr = raw.bus_arr
    num_buses = size(bus_arr,1)
    buses = Array{Bus}(undef, num_buses)
    bus_ref=-1
    for i in 1:num_buses
        @assert bus_arr[i,1]>0  #don't support nonpositive bus ids
        bus_arr[i,9] *= pi/180 # ANIRUDH: Bus is an immutable struct. Modify bus_arr itself
        buses[i] = Bus(bus_arr[i,1:13]...)
        # buses[i].Va *= pi/180 # ANIRUDH: See previous comment
        if buses[i].bustype==3
            if bus_ref>0
                error("More than one reference bus present in the data")
            else 
                bus_ref=i
            end
        end
        #println("bus ", i, " ", buses[i].Vmin, "      ", buses[i].Vmax)
    end

    #
    # load branches/lines
    #
    branch_arr = raw.branch_arr
    num_lines = size(branch_arr,1)
    lines_on = findall((branch_arr[:,11].>0) .& ((branch_arr[:,1].!=lineOff.from) .| (branch_arr[:,2].!=lineOff.to)) )
    num_on   = length(lines_on)

    if lineOff.from>0 && lineOff.to>0 
        # println("opf_loaddata: was asked to remove line from,to=", lineOff.from, ",", lineOff.to)
    end
    if length(findall(branch_arr[:,11].==0))>0
        # println("opf_loaddata: ", num_lines-length(findall(branch_arr[:,11].>0)), " lines are off and will be discarded (out of ", num_lines, ")")
    end



    lines = Array{Line}(undef, num_on)

    lit=0
    for i in lines_on
        @assert branch_arr[i,11] == 1  #should be on since we discarded all other
        lit += 1
        lines[lit] = Line(branch_arr[i, 1:13]...)
        if lines[lit].angmin>-360 || lines[lit].angmax<360
            error("Bounds of voltage angles are still to be implemented.")
        end

    end
    @assert lit == num_on

    #
    # load generators
    #
    gen_arr = raw.gen_arr
    costgen_arr = raw.costgen_arr
    num_gens = size(gen_arr,1)

    baseMVA=100

    @assert num_gens == size(costgen_arr,1)

    gens_on=findall(gen_arr[:,8].!=0); num_on=length(gens_on)
    if num_gens-num_on>0
        println("loaddata: ", num_gens-num_on, " generators are off and will be discarded (out of ", num_gens, ")")
    end

    generators = Array{Gener}(undef, num_on)
    R = 0.04 # Droop regulation
    i=0
    for git in gens_on
        i += 1

        generators[i] = Gener(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Array{Int}(undef, 0),0)

        generators[i].bus      = gen_arr[git,1]
        generators[i].Pg       = gen_arr[git,2] / baseMVA
        generators[i].Qg       = gen_arr[git,3] / baseMVA
        generators[i].Qmax     = gen_arr[git,4] / baseMVA
        generators[i].Qmin     = gen_arr[git,5] / baseMVA
        generators[i].Vg       = gen_arr[git,6]
        generators[i].mBase    = gen_arr[git,7]
        generators[i].status   = gen_arr[git,8]
        @assert generators[i].status==1
        generators[i].Pmax     = gen_arr[git,9]  / baseMVA
        generators[i].Pmin     = gen_arr[git,10] / baseMVA
        generators[i].Pc1      = gen_arr[git,11]
        generators[i].Pc2      = gen_arr[git,12]
        generators[i].Qc1min   = gen_arr[git,13]
        generators[i].Qc1max   = gen_arr[git,14]
        generators[i].Qc2min   = gen_arr[git,15]
        generators[i].Qc2max   = gen_arr[git,16]
        generators[i].ramp_agc = gen_arr[git,9] * ramp_scale  / baseMVA
        generators[i].scen_agc = gen_arr[git,9] * 0.1  / baseMVA
        generators[i].gentype  = costgen_arr[git,1]
        generators[i].startup  = costgen_arr[git,2]
        generators[i].shutdown = costgen_arr[git,3]
        generators[i].n        = costgen_arr[git,4]
        generators[i].alpha    = -((1/R)*generators[i].Pmax)
        if generators[i].gentype == 1
            generators[i].coeff = costgen_arr[git,5:end]  
            error("Piecewise linear costs remains to be implemented.")
        else
            if generators[i].gentype == 2
                generators[i].coeff = costgen_arr[git,5:end]
                #println(generators[i].coeff, " ", length(generators[i].coeff), " ", generators[i].coeff[2])
            else
                error("Invalid generator cost model in the data.")
            end 
        end
    end

    # build a dictionary between buses ids and their indexes
    busIdx = mapBusIdToIdx(buses)

    # set up the FromLines and ToLines for each bus
    FromLines,ToLines = mapLinesToBuses(buses, lines, busIdx)

    # generators at each bus
    BusGeners = mapGenersToBuses(buses, generators, busIdx)

    # demands for multiperiod OPF
    Pd = raw.pd_arr
    Qd = raw.qd_arr
    if time_horizon > 0
        Pd = Pd[:,1:time_horizon] .* load_scale
        Qd = Qd[:,1:time_horizon] .* load_scale
    end

    return OPFData(buses, lines, generators, bus_ref, baseMVA, busIdx, FromLines, ToLines, BusGeners, Pd, Qd)
end

function  computeAdmitances(lines, buses, baseMVA; lossless::Bool=false, fixedtaps::Bool=false, zeroshunts::Bool=false)
    nlines = length(lines)
    YffR=Array{Float64}(undef, nlines)
    YffI=Array{Float64}(undef, nlines)
    YttR=Array{Float64}(undef, nlines) 
    YttI=Array{Float64}(undef, nlines)
    YftR=Array{Float64}(undef, nlines)
    YftI=Array{Float64}(undef, nlines)
    YtfR=Array{Float64}(undef, nlines)
    YtfI=Array{Float64}(undef, nlines)

    for i in 1:nlines
        @assert lines[i].status == 1 
        Ys = 1/((lossless ? 0.0 : lines[i].r) + lines[i].x*im) 
        #assign nonzero tap ratio
        tap = (lines[i].ratio==0) ? 1.0 : lines[i].ratio
        fixedtaps && (tap = 1.0)

        #add phase shifters
        if (!lossless)
            tap *= exp(lines[i].angle * pi/180 * im)
        end

        Ytt = Ys + lines[i].b/2*im
        Yff = Ytt / (tap*conj(tap))
        Yft = -Ys / conj(tap)
        Ytf = -Ys / tap

        #split into real and imag parts
        YffR[i] = real(Yff); YffI[i] = imag(Yff)
        YttR[i] = real(Ytt); YttI[i] = imag(Ytt)
        YtfR[i] = real(Ytf); YtfI[i] = imag(Ytf)
        YftR[i] = real(Yft); YftI[i] = imag(Yft)

        # if lossless
        #   if !iszero(lines[i].r)
        #     println("warning: lossless assumption changes r from ", lines[i].r, " to 0 for line ", lines[i].from, " -> ", lines[i].to)
        #   end
        #   if !iszero(lines[i].angle)
        #     println("warning: lossless assumption changes angle from ", lines[i].angle, " to 0 for line ", lines[i].from, " -> ", lines[i].to)
        #   end
        # end
    end

    nbuses = length(buses)
    YshR = Array{Float64}(undef, nbuses)
    YshI = Array{Float64}(undef, nbuses)
    for i in 1:nbuses
        YshR[i] = (lossless ? 0.0 : (buses[i].Gs / baseMVA))
        YshI[i] = buses[i].Bs / baseMVA
        zeroshunts && (YshI[i] = 0)
        # if lossless && !iszero(buses[i].Gs)
        #   println("warning: lossless assumption changes Gshunt from ", buses[i].Gs, " to 0 for bus ", i)
        # end
    end

    @assert 0==length(findall(isnan.(YffR)))+length(findall(isinf.(YffR)))
    @assert 0==length(findall(isnan.(YffI)))+length(findall(isinf.(YffI)))
    @assert 0==length(findall(isnan.(YttR)))+length(findall(isinf.(YttR)))
    @assert 0==length(findall(isnan.(YttI)))+length(findall(isinf.(YttI)))
    @assert 0==length(findall(isnan.(YftR)))+length(findall(isinf.(YftR)))
    @assert 0==length(findall(isnan.(YftI)))+length(findall(isinf.(YftI)))
    @assert 0==length(findall(isnan.(YtfR)))+length(findall(isinf.(YtfR)))
    @assert 0==length(findall(isnan.(YtfI)))+length(findall(isinf.(YtfI)))
    @assert 0==length(findall(isnan.(YshR)))+length(findall(isinf.(YshR)))
    @assert 0==length(findall(isnan.(YshI)))+length(findall(isinf.(YshI)))
    if lossless
        @assert 0==length(findall(!iszero, YffR))
        @assert 0==length(findall(!iszero, YttR))
        @assert 0==length(findall(!iszero, YftR))
        @assert 0==length(findall(!iszero, YtfR))
        @assert 0==length(findall(!iszero, YshR))
    end

    return YffR, YffI, YttR, YttI, YftR, YftI, YtfR, YtfI, YshR, YshI
end


# Builds a map from lines to buses.
# For each line we store an array with zero or one element containing
# the  'From' and 'To'  bus number. 
function mapLinesToBuses(buses, lines, busDict)
    nbus = length(buses)
    FromLines = [Int[] for b in 1:nbus]
    ToLines   = [Int[] for b in 1:nbus]
    for i in 1:length(lines)
        busID = busDict[lines[i].from]
        @assert 1<= busID <= nbus
        push!(FromLines[busID], i)

        busID = busDict[lines[i].to]
        @assert 1<= busID  <= nbus
        push!(ToLines[busID], i)
    end
    return FromLines,ToLines
end

# Builds a mapping between bus ids and bus indexes
#
# Returns a dictionary with bus ids as keys and bus indexes as values
function mapBusIdToIdx(buses)
    dict = Dict{Int,Int}()
    for b in 1:length(buses)
        @assert !haskey(dict,buses[b].bus_i)
        dict[buses[b].bus_i] = b
    end
    return dict
end


# Builds a map between buses and generators.
# For each bus we keep an array of corresponding generators number (as array).
# 
# (Can be more than one generator per bus)
function mapGenersToBuses(buses, generators,busDict)
    gen2bus = [Int[] for b in 1:length(buses)]
    for g in 1:length(generators)
        busID = busDict[ generators[g].bus ]
        #@assert(0==length(gen2bus[busID])) #at most one generator per bus
        push!(gen2bus[busID], g)
    end
    return gen2bus
end

