using Test, MPI
using ProxAL

testdir = @__DIR__
# MPI is a requirement for ExaTron
MPI.Init()

@testset "Integration tests" begin
    include("blockmodel.jl")
end

# We can finalize here as now we launch external processes
MPI.Finalize()

# Testing using 1 process
@testset "Sequential tests" begin
    mpiexec() do cmd
        run(`$cmd -n 1 $(Base.julia_cmd()) --project=$testdir/.. $testdir/single.jl`)
    end
    @test true
end

# Testing using 2 processes

@testset "Parallel tests" begin
    mpiexec() do cmd
        run(`$cmd -n 2 $(Base.julia_cmd()) --project=$testdir/.. $testdir/distributed.jl`)
    end
    @test true
end
