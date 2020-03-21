using DelimitedFiles
using JuMP
using MathProgBase
using Ipopt
using Printf

include("mpacopf_data.jl")
include("mpacopf_model.jl")

function usage()
    println("Usage: julia mpc.jl case scen T H LS RS warm opt [profname]")
    println(" where")
    println("          case - the name of the case file")
    println("          scen - the name of the scenario file")
    println("             T - the length of the time horizon")
    println("            RS - ramping scale")
end

function main(args)

    # ---------------------------------------------------------------------
    # Parse the arguments.
    # ---------------------------------------------------------------------

    if length(args) < 4
        usage()
        return
    end

    case = args[1]
    scen = args[2]
    load_scale = 1.0

    T = max(parse(Int,args[3]),1)
    ramp_scale = parse(Float64,args[4])

    baseMVA = 100

    println("Options specified:")
    println("        case: ", case)
    println("        scen: ", scen)
    println("           T: ", T)
    println("  ramp scale: ", ramp_scale)
    flush(stdout)

    # ---------------------------------------------------------------------
    # Read the circuit and load.
    # ---------------------------------------------------------------------

    circuit = getcircuit(case, baseMVA, ramp_scale)
    load = getload(scen, load_scale)
    num_gens = length(circuit.gen)
    num_buses = length(circuit.bus)
    gen = circuit.gen

    # ---------------------------------------------------------------------
    # Solve the first time horizon [1:T] using Ipopt.
    # ---------------------------------------------------------------------

    demand = Load(load.pd[:,1:T], load.qd[:,1:T])
    m_cur = get_mpmodel(circuit, demand)
    init_x(m_cur, circuit, demand)
    setsolver(m_cur, IpoptSolver(option_file_name="ipopt.opt"))
    stat = solve(m_cur)

    if stat != :Optimal
        println("Stat is not optimal: ", stat)
        return
    end
end

main(ARGS)
