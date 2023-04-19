
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

            status, minor_iter = ExaTron.ExaTronCUDAKernels.tron_qp_kernel(n, 500, 200, 1e-6, 1.0, x, xl, xu, A, c)

            u[pg_idx] = x[1]
            s[I] = x[2]
        end
    end

    return
end

function generator_kernel_two_level(
    model::ModelProxAL{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1}
)

    n = 2
    shmem_size = sizeof(Float64)*(14*n+3+3*n^2) + sizeof(Int)*(3*n+3)
    ngen = model.grid_data.ngen

    tgpu = CUDA.@timed @cuda threads=32 blocks=ngen shmem=shmem_size generator_kernel_two_level_proxal(
                ngen, model.gen_start,
                u, xbar, zu, lu, rho_u,
                model.grid_data.pgmin, model.grid_data.pgmax,
                model.grid_data.qgmin, model.grid_data.qgmax,
                model.smin, model.smax, model.s_curr,
                model.Q_ref, model.c_ref)
    return tgpu
end

@kernel function generator_kernel_two_level_proxal_ka(ngen::Int, gen_start::Int,
    u, xbar, z,
    l, rho,
    pgmin, pgmax,
    qgmin, qgmax,
    smin, smax, s,
    _A, _c)

    tx = @index(Local, Linear)
    I = @index(Group, Linear)

    n = 2

    if I <= ngen
        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)

        A = @localmem Float64 (n,n)
        c = @localmem Float64 (n,)

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
        @synchronize

        @inbounds begin
            xl[1] = pgmin[I]
            xu[1] = pgmax[I]
            xl[2] = smin[I]
            xu[2] = smax[I]
            x[1] = min(xu[1], max(xl[1], u[pg_idx]))
            x[2] = min(xu[2], max(xl[2], s[I]))
            @synchronize

            status, minor_iter = ExaTron.ExaTronKAKernels.tron_qp_kernel(n, 500, 200, 1e-6, 1.0, x, xl, xu, A, c, tx)

            u[pg_idx] = x[1]
            s[I] = x[2]
        end
    end
end

function generator_kernel_two_level(
    model::ModelProxAL{Float64,AT,IAT},
    baseMVA::Float64, u::AT, xbar::AT,
    zu::AT, lu::AT, rho_u::AT, device
) where {AT, IAT}

    ngen = model.grid_data.ngen

    #wait(driver_kernel_test(device,n)(Val{n}(),max_feval,max_minor,dx,dxl,dxu,dA,dc,d_out,ndrange=(n,nblk),dependencies=Event(device)))
    # tgpu = CUDA.@timed @cuda threads=32 blocks=ngen shmem=shmem_size generator_kernel_two_level_proxal(
# @kernel function generator_kernel_two_level_proxal(ngen::Int, gen_start::Int,
    generator_kernel_two_level_proxal_ka(device, 32)(
        32, model.gen_start,
        u, xbar, zu, lu, rho_u,
        model.grid_data.pgmin, model.grid_data.pgmax,
        model.grid_data.qgmin, model.grid_data.qgmax,
        model.smin, model.smax, model.s_curr,
        model.Q_ref, model.c_ref,
        ndrange=(ngen,ngen)
    )
    KA.synchronize(device)
    return 0.0
end
