
slack_values(model::ModelProxAL) = model.s_curr

# SETTERS
macro define_setter_array(function_name, attribute)
    fname = Symbol(function_name)
    quote
        function $(esc(fname))(model::ModelProxAL, values::AbstractVector)
            copyto!(model.$attribute, values)
            return
        end
    end
end

macro define_setter_value(function_name, attribute)
    fname = Symbol(function_name)
    quote
        function $(esc(fname))(model::ModelProxAL, value::AbstractFloat)
            model.$attribute = value
            return
        end
    end
end

function set_active_load!(model::ModelProxAL, values::AbstractVector)
    copyto!(model.grid_data.Pd, values)
end
function set_reactive_load!(model::ModelProxAL, values::AbstractVector)
    copyto!(model.grid_data.Qd, values)
end
function set_generator_cost!(model::ModelProxAL)
    ngen = model.grid_data.ngen
    baseMVA = model.grid_data.baseMVA
    Q_ref = zero(model.Q_ref)
    c_ref = zero(model.c_ref)
    Q_ref[1:4:4*ngen] .= 2.0 * model.grid_data.c2[:] * baseMVA^2
    c_ref[1:2:2*ngen] .= model.grid_data.c1[:] * baseMVA
    copyto!(model.Q_ref, Q_ref)
    copyto!(model.c_ref, c_ref)
end

#=
    GETTERS
=#

function active_power_generation(model::ModelProxAL, sol::ExaAdmm.Solution)
    ngen = model.grid_data.ngen
    return sol.u_curr[1:2:2*ngen]
end
function reactive_power_generation(model::ModelProxAL, sol::ExaAdmm.Solution)
    ngen = model.grid_data.ngen
    return sol.u_curr[2:2:2*ngen]
end
function voltage_magnitude(model::ModelProxAL, sol::ExaAdmm.Solution)
    data, line_start = model.grid_data, model.line_start
    nbus    = data.nbus
    FrIdx   = data.FrIdx   |> Array
    FrStart = data.FrStart |> Array
    ToIdx   = data.ToIdx   |> Array
    ToStart = data.ToStart |> Array
    v_curr  = sol.v_curr   |> Array

    vm = zeros(nbus)
    for I=1:nbus
        for j=FrStart[I]:FrStart[I+1]-1
            pij_idx = line_start + 8*(FrIdx[j]-1)
            vm[I] = sqrt(abs(v_curr[pij_idx+4]))
        end
        for j=ToStart[I]:ToStart[I+1]-1
            pij_idx = line_start + 8*(ToIdx[j]-1)
            vm[I] = sqrt(abs(v_curr[pij_idx+5]))
        end
    end
    return vm
end
function voltage_angle(model::ModelProxAL, sol::ExaAdmm.Solution)
    data, line_start = model.grid_data, model.line_start
    data, line_start = model.grid_data, model.line_start
    nbus    = data.nbus
    FrIdx   = data.FrIdx   |> Array
    FrStart = data.FrStart |> Array
    ToIdx   = data.ToIdx   |> Array
    ToStart = data.ToStart |> Array
    v_curr  = sol.v_curr   |> Array

    va = zeros(nbus)
    for I=1:nbus
        for j=FrStart[I]:FrStart[I+1]-1
            pij_idx = line_start + 8*(FrIdx[j]-1)
            va[I] = v_curr[pij_idx+6]
        end
        for j=ToStart[I]:ToStart[I+1]-1
            pij_idx = line_start + 8*(ToIdx[j]-1)
            va[I] = v_curr[pij_idx+7]
        end
    end
    return va
end


active_power_generation(model::ModelProxAL) = active_power_generation(model, model.solution)
reactive_power_generation(model::ModelProxAL) = reactive_power_generation(model, model.solution)


#=
    ExaAdmm
=#
function ExaAdmm.acopf_admm_update_x_gen(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelProxAL{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    gen_sol::ExaAdmm.EmptyGeneratorSolution{Float64,Array{Float64,1}}
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    time_gen = generator_kernel_two_level(mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time
    return
end

function ExaAdmm.acopf_admm_update_x_gen(
    env::ExaAdmm.AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelProxAL{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    gen_sol::ExaAdmm.EmptyGeneratorSolution{Float64,CuArray{Float64,1}}
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    time_gen = generator_kernel_two_level(mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time
    return
end

function ExaAdmm.acopf_admm_update_x_gen(
    env::ExaAdmm.AdmmEnv,
    mod::ModelProxAL,
    gen_sol::ExaAdmm.EmptyGeneratorSolution,
    device
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    generator_kernel_two_level(mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho, device)
    info.user.time_generators += 0.0
    info.time_x_update += 0.0
    return
end
