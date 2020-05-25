using Distributed
using SharedArrays
using JuMP, Ipopt
N=1000
ret = SharedVector{Float64}(N)
@sync @distributed  for i=1:N
    function f(v)
        m = JuMP.Model(JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 1))
        @variable(m, x[1:1000] >= i/N)
        @objective(m, Min, sum(x[i]^2 for i=1:1000))
        JuMP.optimize!(m)
        v[i] = JuMP.objective_value(m)
        return 0.0
    end
    f(ret)
end

@show(ret)
