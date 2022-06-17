using Test, MPI
using ProxAL

testdir = @__DIR__

@testset "ProxAL" begin
    @testset "Integration tests" begin
        include("blockmodel.jl")
    end
    @testset "ExaAdmm backend" begin
        include("exaadmm.jl")
    end

    # We can finalize here as now we launch external processes

    # Testing using 1 process
    @testset "Sequential tests" begin
        include("convergence.jl")
    end

    # Testing using 2 processes

    @testset "Parallel tests" begin
        mpiexec() do cmd
            run(`$cmd -n 2 $(Base.julia_cmd()) --project=$testdir/.. $testdir/convergence.jl 1`)
        end
        @test true
    end
end
