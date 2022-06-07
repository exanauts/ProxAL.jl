
function generator_kernel_two_level_proxal(ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, xbar::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    smin::CuDeviceArray{Float64,1}, smax::CuDeviceArray{Float64,1}, s::CuDeviceArray{Float64,1},
    _A::CuDeviceArray{Float64,1}, _c::CuDeviceArray{Float64,1})

    tx = threadIdx().x
    I = blockIdx().x

    if I <= ngen
        n = 2
        x = @cuDynamicSharedMem(Float64, n)
        xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
        xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))

        A = @cuDynamicSharedMem(Float64, (n,n), (13*n+3)*sizeof(Float64)+(3*n+3)*sizeof(Int))
        c = @cuDynamicSharedMem(Float64, n, (13*n+3+3*n^2)*sizeof(Float64)+(3*n+3)*sizeof(Int))

        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-xbar[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        A_start = 4*(I-1)
        c_start = 2*(I-1)
        if tx <= n
            @inbounds begin
                for j=1:n
                    A[tx,j] = _A[n*(j-1)+tx + A_start]
                end
                c[tx] = _c[tx + c_start]

                if tx == 1
                    A[1,1] += rho[pg_idx]
                    c[1] += l[pg_idx] + rho[pg_idx]*(-xbar[pg_idx] + z[pg_idx])
                end
            end
        end
        CUDA.sync_threads()

        @inbounds begin
            xl[1] = pgmin[I]
            xu[1] = pgmax[I]
            xl[2] = smin[I]
            xu[2] = smax[I]
            x[1] = min(xu[1], max(xl[1], u[pg_idx]))
            x[2] = min(xu[2], max(xl[2], s[I]))
            CUDA.sync_threads()

            status, minor_iter = ExaTron.tron_qp_kernel(n, 500, 200, 1e-6, 1.0, x, xl, xu, A, c)

            u[pg_idx] = x[1]
            s[I] = x[2]
        end
    end

    return
end

function _build_qp_kernel!(
    Q::CuDeviceArray{Float64, 1}, c::CuDeviceArray{Float64, 1}, t_curr, T,
    Q_ref::CuDeviceArray{Float64, 1}, c_ref::CuDeviceArray{Float64, 1},
    baseMVA::Float64, c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1},
    tau_prox::Float64, rho_prox::Float64,
    pg_ref_prox::CuDeviceArray{Float64,1},
    l_next_prox::CuDeviceArray{Float64,1}, pg_next_prox::CuDeviceArray{Float64,1},
    l_prev_prox::CuDeviceArray{Float64,1}, pg_prev_prox::CuDeviceArray{Float64,1},
    ngen::Int,
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if (I <= ngen)
        shift_Q = 4*(I-1)
        shift_c = 2*(I-1)

        # Q[1, 1]
        Q[shift_Q + 1] = Q_ref[shift_Q + 1]
        # Q[2, 1]
        Q[shift_Q + 2] = Q_ref[shift_Q + 2]
        # Q[1, 2]
        Q[shift_Q + 3] = Q_ref[shift_Q + 3]
        # Q[2, 2]
        Q[shift_Q + 4] = Q_ref[shift_Q + 4]

        # c[1]
        c[shift_c + 1] = c_ref[shift_c + 1]
        # c[2]
        c[shift_c + 2] = c_ref[shift_c + 2]

        # τ/2 (pg - pgₚ)^2
        Q[shift_Q + 1] += tau_prox + 2.0 * c2[I]*baseMVA^2
        c[shift_c + 1] += - tau_prox * pg_ref_prox[I] + c1[I]*baseMVA

        # λ₊ (pg - pg₊) + ρ/2 * (pg - pg₊)^2
        if (t_curr < T)
            # Pg
            Q[shift_Q + 1] += rho_prox
            c[shift_c + 1] += l_next_prox[I]
            c[shift_c + 1] -= rho_prox * pg_next_prox[I]
        end

        # λ₋ (pg₋ + s - pg) + ρ/2 * (pg₋ + s - pg)^2
        if (t_curr > 1)
            # Pg
            Q[shift_Q + 1] += rho_prox
            c[shift_c + 1] -= l_prev_prox[I]
            c[shift_c + 1] -= rho_prox * pg_prev_prox[I]

            # Slack
            Q[shift_Q + 4] += rho_prox
            c[shift_c + 2] += l_prev_prox[I]
            c[shift_c + 2] += rho_prox * pg_prev_prox[I]

            # Cross-term (Slack * Pg)
            Q[shift_Q + 2] -= rho_prox
            Q[shift_Q + 3] -= rho_prox
        end
    end
    return
end

function build_qp_problem!(baseMVA, model::ModelProxAL{Float64, CuVector{Float64}})
    ngen = model.grid_data.ngen
    nblk_gen = div(ngen, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_gen _build_qp_kernel!(
        model.Q, model.c, model.t_curr, model.T,
        model.Q_ref, model.c_ref,
        baseMVA, model.grid_data.c2, model.grid_data.c1,
        model.tau, model.rho, model.pg_ref,
        model.l_next, model.pg_next,
        model.l_prev, model.pg_prev, ngen,
    )
    return tgpu
end

function generator_kernel_two_level(
    model::ModelProxAL{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1})

    n = 2
    shmem_size = sizeof(Float64)*(14*n+3+3*n^2) + sizeof(Int)*(3*n+3)
    ngen = model.grid_data.ngen

    tgpu1 = build_qp_problem!(baseMVA, model)
    tgpu2 = CUDA.@timed @cuda threads=32 blocks=ngen shmem=shmem_size generator_kernel_two_level_proxal(
                ngen, model.gen_start,
                u, xbar, zu, lu, rho_u,
                model.grid_data.pgmin, model.grid_data.pgmax,
                model.grid_data.qgmin, model.grid_data.qgmax,
                model.smin, model.smax,
                model.s_curr, model.Q, model.c)
    return tgpu2
end

