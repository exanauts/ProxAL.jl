function generator_kernel_two_level_proxal(ngen::Int, gen_start::Int,
    u, xbar, z, l, rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1}, qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    smin::Array{Float64,1}, smax::Array{Float64,1}, s::Array{Float64,1},
    _A::Array{Float64,1}, _c::Array{Float64,1})

    n = 2
    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)
    A = zeros(n,n)
    c = zeros(n)

    @inbounds for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-xbar[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        A_start = 4*(I-1)
        c_start = 2*(I-1)
        for i=1:n
            for j=1:n
                A[i,j] = _A[n*(j-1)+i + A_start]
            end
            c[i] = _c[i + c_start]
        end

        A[1,1] += rho[pg_idx]
        c[1] += l[pg_idx] + rho[pg_idx]*(-xbar[pg_idx] + z[pg_idx])

        xl[1] = pgmin[I]
        xu[1] = pgmax[I]
        xl[2] = smin[I]
        xu[2] = smax[I]
        x[1] = min(xu[1], max(xl[1], u[pg_idx]))
        x[2] = min(xu[2], max(xl[2], s[I]))

        function eval_f_cb(x)
            f = 0.0
            for j=1:n
                for i=1:n
                    f += x[i]*A[i,j]*x[j]
                end
            end
            f *= 0.5
            for i=1:n
                f += c[i]*x[i]
            end
            return f
        end

        function eval_g_cb(x, g)
            for i=1:n
                g[i] = 0.0
                for j=1:n
                    g[i] += A[i,j]*x[j]
                end
                g[i] += c[i]
            end
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            if mode == :Structure
                rows[1] = 1; cols[1] = 1
                rows[2] = 2; cols[2] = 1
                rows[3] = 2; cols[3] = 2
            else
                values[1] = A[1,1]
                values[2] = A[2,1]
                values[3] = A[2,2]
            end
            return
        end

        nele_hess = 3
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => 1e-6, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)

        tron.x .= x
        status = ExaTron.solveProblem(tron)

        u[pg_idx] = tron.x[1]
        s[I] = tron.x[2]
    end

    return
end

function build_qp_problem!(baseMVA, model::ModelProxAL)
    for g in 1:model.grid_data.ngen
        shift_Q = 4*(g-1)
        shift_c = 2*(g-1)

        # Q[1, 1]
        model.Q[shift_Q + 1] = model.Q_ref[shift_Q + 1]
        # Q[2, 1]
        model.Q[shift_Q + 2] = model.Q_ref[shift_Q + 2]
        # Q[1, 2]
        model.Q[shift_Q + 3] = model.Q_ref[shift_Q + 3]
        # Q[2, 2]
        model.Q[shift_Q + 4] = model.Q_ref[shift_Q + 4]

        # c[1]
        model.c[shift_c + 1] = model.c_ref[shift_c + 1]
        # c[2]
        model.c[shift_c + 2] = model.c_ref[shift_c + 2]

        # τ/2 (pg - pgₚ)^2
        model.Q[shift_Q + 1] += model.tau + 2.0 * model.grid_data.c2[g]*baseMVA^2
        model.c[shift_c + 1] += - model.tau * model.pg_ref[g] + model.grid_data.c1[g]*baseMVA

        # λ₊ (pg - pg₊) + ρ/2 * (pg - pg₊)^2
        if (model.t_curr < model.T)
            # Pg
            model.Q[shift_Q + 1] += model.rho
            model.c[shift_c + 1] += model.l_next[g]
            model.c[shift_c + 1] -= model.rho * model.pg_next[g]
        end

        # λ₋ (pg₋ + s - pg) + ρ/2 * (pg₋ + s - pg)^2
        if (model.t_curr > 1)
            # Pg
            model.Q[shift_Q + 1] += model.rho
            model.c[shift_c + 1] -= model.l_prev[g]
            model.c[shift_c + 1] -= model.rho * model.pg_prev[g]

            # Slack
            model.Q[shift_Q + 4] += model.rho
            model.c[shift_c + 2] += model.l_prev[g]
            model.c[shift_c + 2] += model.rho * model.pg_prev[g]

            # Cross-term (Slack * Pg)
            model.Q[shift_Q + 2] -= model.rho
            model.Q[shift_Q + 3] -= model.rho
        end
    end
end

function generator_kernel_two_level(
    model::ModelProxAL{Float64,Array{Float64,1},Array{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u,
)
    build_qp_problem!(baseMVA, model)
    tcpu = @timed generator_kernel_two_level_proxal(
        model.grid_data.ngen, model.gen_start,
        u, xbar, zu, lu, rho_u,
        model.grid_data.pgmin, model.grid_data.pgmax,
        model.grid_data.qgmin, model.grid_data.qgmax,
        model.smin, model.smax, model.s_curr, model.Q, model.c)
    return tcpu
end

