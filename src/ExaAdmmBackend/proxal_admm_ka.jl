@kernel function generator_kernel_two_level_proxal_ka(ngen::Int, gen_start::Int,
    u, xbar, z,
    l, rho,
    @Const(pgmin), @Const(pgmax),
    @Const(qgmin), @Const(qgmax),
    @Const(smin), @Const(smax), s,
    _A, _c)

    tx = @index(Local, Linear)
    I = @index(Group, Linear)
    # grpsize = @groupsize
    # range = @ndrange
    # AMDGPU.@rocprintf "generator_kernel %s %s %s\n" ngen grpsize range
    x = @localmem Float64 (2,)
    xl = @localmem Float64 (2,)
    xu = @localmem Float64 (2,)

    A = @localmem Float64 (2,2)
    c = @localmem Float64 (2,)

    if I <= ngen
        @synchronize


        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-xbar[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        A_start = 4*(I-1)
        c_start = 2*(I-1)
        if tx <= 2
            # @inbounds begin
                for j=1:2
                    A[tx,j] = _A[2*(j-1)+tx + A_start]
                end
                c[tx] = _c[tx + c_start]

                if tx == 1
                    A[1,1] += rho[pg_idx]
                    c[1] += l[pg_idx] + rho[pg_idx]*(-xbar[pg_idx] + z[pg_idx])
                end
            # end
        end
        @synchronize

        # @inbounds begin
            if tx == 1
                xl[1] = pgmin[I]
                xu[1] = pgmax[I]
                xl[2] = smin[I]
                xu[2] = smax[I]
                x[1] = min(xu[1], max(xl[1], u[pg_idx]))
                x[2] = min(xu[2], max(xl[2], s[I]))
            end
            @synchronize
            # AMDGPU.@rocprintf "btron_qp_kernel: %s < %s < %s %s < %s < %s \n" xl[1] x[1] xu[1] xl[2] x[2] xu[2]
            status, minor_iter = ExaTron.ExaTronKAKernels.tron_qp_kernel(2, 500, 200, 1e-6, 1.0, x, xl, xu, A, c, tx)
            @synchronize
            if tx == 1
                u[pg_idx] = x[1]
                s[I] = x[2]
            end
            @synchronize
        # end
    end
end

function generator_kernel_two_level(
    model::ModelProxAL{Float64,AT,IAT},
    baseMVA::Float64, u::AT, xbar::AT,
    zu::AT, lu::AT, rho_u::AT, device
) where {AT, IAT}

    ngen = model.grid_data.ngen
    println("Pgmin/Pgmax: $(model.grid_data.pgmin), $(model.grid_data.pgmax)")
    println("smin/smax: $(model.smin), $(model.smax)")
    println("device: $device")
    println("typeof: $(typeof(model.grid_data.pgmin)) $(typeof(model.smin))")
    KA.synchronize(device)
    generator_kernel_two_level_proxal_ka(device, 2, 2*ngen)(
        ngen, model.gen_start,
        u, xbar, zu, lu, rho_u,
        model.grid_data.pgmin, model.grid_data.pgmax,
        model.grid_data.qgmin, model.grid_data.qgmax,
        model.smin, model.smax, model.s_curr,
        model.Q_ref, model.c_ref,
    )
    KA.synchronize(device)
    println("u: $u")
    println("s: $(model.s_curr)")
    println("Synchronize done")
    return 0.0
end
