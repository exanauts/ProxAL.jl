include("acopf.jl")
include("acopf_socp.jl")
include("acopf_sdp.jl")


using Ipopt
using SCS


function acopf_main()
    opfdata = opf_loaddata(ARGS[1])
    network = buildNetwork(opfdata.buses, opfdata.lines, opfdata.BusIdx)
    buildNetworkPartition(network, num_partitions=3)

    acopf_solve_aladin(opfdata, network)
end


using TimerOutputs
global oldJuMPVersion = (Pkg.installed()["JuMP"] <= v"0.18.6")
const TL = 1000.0

if oldJuMPVersion

else
    using Mosek, MosekTools
    using COSMO
end

function test(opfdata, to, str)
    if oldJuMPVersion
        @timeit to "ACOPF - Ipopt" testAcopf(opfdata, IpoptSolver(print_level = 1, max_cpu_time = TL), (str == nothing) ? nothing : "acopf ipopt")
        @timeit to "SOCP  - Ipopt" testSocp(opfdata, IpoptSolver(print_level = 1, max_cpu_time = TL), false, (str == nothing) ? nothing : " socp ipopt")
        @timeit to "SOCP  - SCS"   testSocp(opfdata, SCSSolver(verbose = 1, max_iters = 100000), true, (str == nothing) ? nothing : " socp   scs")
        @timeit to "SDP   - SCS"   testSdp(opfdata, SCSSolver(verbose = 1, max_iters = 100000), (str == nothing) ? nothing : "  sdp   scs")
    else
        #@timeit to "SOCP  - COSMO" testSocp(opfdata, with_optimizer(COSMO.Optimizer, verbose = true, max_iter = 100000, time_limit = TL), true, (str == nothing) ? nothing : " socp cosmo")
        @timeit to "SOCP  - Mosek" testSocp(opfdata, with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG = 0, MSK_DPAR_OPTIMIZER_MAX_TIME = TL), true, (str == nothing) ? nothing : " socp mosek")
        @timeit to "SDP   - COSMO" testSdp(opfdata, with_optimizer(COSMO.Optimizer, verbose = true, max_iter = 100000, time_limit = TL), (str == nothing) ? nothing : "  sdp cosmo")
        #@timeit to "SDP   - Mosek" testSdp(opfdata, with_optimizer(Mosek.Optimizer, MSK_IPAR_LOG = 1, MSK_DPAR_OPTIMIZER_MAX_TIME = TL), (str == nothing) ? nothing : "  sdp mosek")
    end
end

function testAcopf(opfdata, solver, str)
    acopfmodel = acopf_model(opfdata, solver)
    timestamp = time()
    acopfmodel, status = acopf_solve(acopfmodel, opfdata)
    solvetime = time() - timestamp
    objval = 0
    if (oldJuMPVersion ? (status == :Optimal) : (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED))
        objval = acopf_outputAll(acopfmodel, opfdata)
    else
        println("could not solve model to optimality! status = ", status)
    end
    if str != nothing
        writeToFile(str, objval, solvetime, status)
    end
end

function testSocp(opfdata, solver, use_conic_interface, str)
    socpmodel = acopfSocp_model(opfdata, solver, use_conic_interface)
    timestamp = time()
    socpmodel, status = acopfSocp_solve(socpmodel, opfdata)
    solvetime = time() - timestamp
    objval = 0
    if (oldJuMPVersion ? (status == :Optimal) : (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.SLOW_PROGRESS))
        objval = acopfSocp_outputAll(socpmodel, opfdata)
        if !oldJuMPVersion && status == MOI.SLOW_PROGRESS
            status = primal_status(socpmodel)
        end
    else
        println("could not solve model to optimality! status = ", status)
    end
    if str != nothing
        writeToFile(str, objval, solvetime, status)
    end
end

function testSdp(opfdata, solver, str)
    sdpmodel = acopfSdp_model(opfdata, solver)
    timestamp = time()
    sdpmodel, status = acopfSdp_solve(sdpmodel, opfdata)
    solvetime = time() - timestamp
    objval = 0
    if (oldJuMPVersion ? (status == :Optimal) : (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.SLOW_PROGRESS))
        objval = acopfSdp_outputAll(sdpmodel, opfdata)
        if !oldJuMPVersion && status == MOI.SLOW_PROGRESS
            status = primal_status(sdpmodel)
        end
    else
        println("could not solve model to optimality! status = ", status)
    end
    if str != nothing
        writeToFile(str, objval, solvetime, status)
    end
end

const t1 = TimerOutput()
opfdata = opf_loaddata("data/case9")
test(opfdata, t1, nothing)

const t2 = TimerOutput()
case = ARGS[1]
case = case[6:end]

function writeToFile(str, objval, solvetime, status)
    open("results/" * case * "-" * (oldJuMPVersion ? "old" : "new") * ".txt", "a+") do f
        @printf(f, "%s %16s %12.2f %12.4f %s\n", str, case, objval, solvetime, status)
    end
end


test(opfdata, t2, "results/" * case)
println(t2)
