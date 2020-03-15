using Random
using Printf

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
    D::Float64 # load characteristic
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

struct Gen
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
    # .gencost fields
    gentype::Int
    startup::Float64
    shutdown::Float64
    n::Int
    coeff2::Float64
    coeff1::Float64
    coeff0::Float64
    alpha::Float64 # generation characteristic
end

struct Yline
    from::Int
    to::Int
    YffR::Float64
    YffI::Float64
    YttR::Float64
    YttI::Float64
    YtfR::Float64
    YtfI::Float64
    YftR::Float64
    YftI::Float64
end

struct Ybus
    bus::Int
    YshR::Float64
    YshI::Float64
end

struct Circuit
    baseMVA::Float64
    busref::Int
    bus::Array{Bus}
    line::Array{Line}
    gen::Array{Gen}
    yline::Array{Yline}
    ybus::Array{Ybus}
    busdict::Dict{Int,Int}
    frombus::Array
    tobus::Array
    bus2gen::Array
end

struct Load
    pd
    qd
end

mutable struct Option
    has_ramping::Bool

    function Option()
        new(true)
    end
end

function get_busmap(bus)
    busdict = Dict{Int,Int}()

    for i in 1:length(bus)
        @assert !haskey(busdict,bus[i].bus_i)
        busdict[bus[i].bus_i] = i
    end

    return busdict
end

function get_linetobusmap(bus, line, busdict)
    num_buses = length(bus)
    from = [Int[] for i in 1:num_buses]
    to = [Int[] for i in 1:num_buses]

    for i in 1:length(line)
        idx = busdict[line[i].from]
        @assert 1 <= idx <= num_buses
        push!(from[idx], i)

        idx = busdict[line[i].to]
        @assert 1 <= idx <= num_buses
        push!(to[idx], i)
    end

    return from, to
end

function get_bustogenmap(bus, gen, busdict)
    bus2gen = [Int[] for i in 1:length(bus)]

    for i in 1:length(gen)
        idx = busdict[gen[i].bus]
        push!(bus2gen[idx], i)
    end

    return bus2gen
end

# -------------------------------------------------------------------------
# Compute admittances.
# -------------------------------------------------------------------------
function getY(case, line, bus, baseMVA)
    dim1 = size(line,1)
    Ys = complex(zeros(dim1, 1))
    tap = complex(ones(dim1, 1))
    Ytt = complex(zeros(dim1, 1))
    Yff = complex(zeros(dim1, 1))
    Yft = complex(zeros(dim1, 1))
    Ytf = complex(zeros(dim1, 1))

    # ---------------------------------------------------------------------
    # bus f: tap bus, bus t: impedance bus or Z bus
    #
    # Ys: the admittance between bus f and bus t.
    #     It is the reciprocal of a series impedance between bus f and t.
    #
    # When there is a off-nominal transformer, the admittance matrix is
    # defined as follows:
    #
    #   / Ift \ = / Yff  Yft \ / Vf \
    #   \ Itf / = \ Ytf  Ytt / \ Vt /
    #
    # where
    #
    #    Yff = ((Ys + j*bft/2) / |a|^2)
    #    Yft = (-Ys / conj(a))
    #    Ytf = (-Ys / a)
    #    Ytt = (Ys + j*bft/2)
    #
    # When we compute If or It (total current injection at bus f or t),
    # we need to add the bus shunt, YshR and YshI.
    # ---------------------------------------------------------------------

    Ys = 1 ./ (line[:,3] .+ line[:,4] .* im)  # r
    ind = findall(line[:,9] .!= 0)
    tap[ind] = line[ind,9]                    # ratio
    tap .*= exp.(line[:,10] .* pi/180 .* im)  # angle
    Ytt = Ys .+ line[:,5] ./ 2 .* im          # b
    Yff = Ytt ./ (tap.*conj.(tap))
    Yft = -Ys ./ conj.(tap)
    Ytf = -Ys ./ tap

    yline = zeros(dim1, 10)
    yline[:,1:2] = line[:,1:2]
    yline[:,3] = real.(Yff); yline[:,4] = imag.(Yff)
    yline[:,5] = real.(Ytt); yline[:,6] = imag.(Ytt)
    yline[:,7] = real.(Ytf); yline[:,8] = imag.(Ytf)
    yline[:,9] = real.(Yft); yline[:,10] = imag.(Yft)

    ybus = zeros(size(bus,1), 3)
    YshR = bus[:,5] ./ baseMVA      # Gs
    YshI = bus[:,6] ./ baseMVA      # Bs
    ybus = [ bus[:,1] YshR YshI ]

    @assert 0==length(findall(isnan.(yline[:,3:10])))
    @assert 0==length(findall(isinf.(yline[:,3:10])))
    @assert 0==length(findall(isnan.(ybus[:,2:3])))
    @assert 0==length(findall(isinf.(ybus[:,2:3])))

    nylines = size(yline,1)
    nybuses = size(ybus,1)
    Ylines = Array{Yline}(undef, nylines)
    Ybuses = Array{Ybus}(undef, nybuses)

    for i in 1:nylines
        Ylines[i] = Yline(yline[i,1:end]...)
    end

    for i in 1:nybuses
        Ybuses[i] = Ybus(ybus[i,1:end]...)
    end

    return Ylines, Ybuses
end

# -------------------------------------------------------------------------
# Get circuit and power demand data for OPF computation.
# -------------------------------------------------------------------------
function getcircuit(case, baseMVA, ramp_scaling)
    bus_mat = readdlm(case*".bus")
    bus_mat[:,9] *= pi/180  # multiply Va by pi/180.
    bus_mat = [ bus_mat zeros(size(bus_mat,1), 1) ]
    bus_mat[:,14] .= rand(1:0.1:1.5) # load characteristic
    # bus_mat[:,14] .= 0

    branch_mat = readdlm(case*".branch")
    active_line_ind = findall(branch_mat[:,11] .> 0)
    line_mat = branch_mat[active_line_ind,:]
    gen_mat = readdlm(case*".gen")
    gencost_mat = readdlm(case*".gencost")

    # Allocate space for coefficients of a quadratic objective function.
    dim1 = size(gen_mat,1)
    gen_mat = [ gen_mat zeros(dim1, 3) zeros(dim1, 1) ]

    # Adjust values based on baseMVA.
    gen_mat[:,[2 3 4 5 9 10]] /= baseMVA  # Pg, Qg, Qmax, Qmin, Pmax, Pmin
    gen_mat[:,17] = gen_mat[:,9] * ramp_scaling  # ramp_agc
    gen_mat[:,18:21] = gencost_mat[:,1:4] # gentype, startup, shutdown, n

    # Set coefficients only for the gentype being 2 (polynomial).
    i = findall((gen_mat[:,18] .== 2))
    @assert(length(i) == dim1)
    gen_mat[i,22:24] = gencost_mat[i,5:7]        # coeff2, coeff1, coeff0

    R = rand(0.04:0.01:0.09) # droop regulation: random number between 4 and 9. R is in p.u.
    gen_mat[:, 25] = -1 ./ (R*(baseMVA / gen_mat[:, 9])) # convert in p.u.

    # Compute admittances.
    yline, ybus = getY(case, line_mat, bus_mat, baseMVA)

    num_buses = size(bus_mat,1)
    num_lines = size(line_mat,1)
    num_gens = size(gen_mat,1)

    bus = Array{Bus}(undef, num_buses)
    line = Array{Line}(undef, num_lines)
    gen = Array{Gen}(undef, num_gens)

    busref = -1
    for i in 1:num_buses
        bus[i] = Bus(bus_mat[i,1:end]...)
        if bus[i].bustype == 3
            if busref > 0
                error("More than one reference bus present")
            else
                busref = i
            end
        end
    end

    for i in 1:num_lines
        line[i] = Line(line_mat[i,1:end]...)
    end

    for i in 1:num_gens
        gen[i] = Gen(gen_mat[i,1:end]...)
    end

    # Create dictionaries due to the lack of set data structure in JuMP.
    busdict = get_busmap(bus)
    frombus, tobus = get_linetobusmap(bus, line, busdict)
    bus2gen = get_bustogenmap(bus, gen, busdict)

    circuit = Circuit(baseMVA, busref, bus, line, gen, yline, ybus,
                      busdict, frombus, tobus, bus2gen)

    return circuit
end

function getload(scen, load_scale)
    pd_mat = readdlm(scen*".Pd")
    qd_mat = readdlm(scen*".Qd")

    load = Load(pd_mat.*load_scale, qd_mat.*load_scale)

    return load
end
