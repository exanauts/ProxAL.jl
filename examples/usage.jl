using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "case"
            help = "Case name [case9, case30, case118, case1354pegase, case2383wp, case9241pegase]"
            required = true
            arg_type = String
            range_tester = x -> x in ["case9", "case30", "case118", "case1354pegase", "case2383wp", "case9241pegase"]
        "--T"
            help = "No. of time periods"
            arg_type = Int
            default = 10
            range_tester = x -> x >= 1
            metavar = "T"
        "--Ctgs"
            help = "No. of line ctgs"
            arg_type = Int
            default = 0
            range_tester = x -> x >= 0
            metavar = "CTGS"
        "--time_unit"
            help = "Select: [hour, minute]"
            default = "minute"
            range_tester = x -> x in ["hour", "minute"]
            metavar = "UNIT"
        "--ramp_value"
            help = "Ramp value: % Pg_max/time_unit"
            arg_type = Float64
            default = 0.5
            range_tester = x -> x >= 0
            metavar = "RVAL"
        "--decompCtgs"
            help = "Decompose contingencies"
            action = :store_true
        "--ramp_constr"
            help = "Select: [penalty, equality, inequality]"
            arg_type = String
            default = "penalty"
            range_tester = x -> x in ["penalty", "equality", "inequality"]
            metavar = "RCON"
        "--Ctgs_constr"
            help = "Select: [frequency_ctrl, preventive_penalty, preventive_equality, corrective_penalty, corrective_equality, corrective_inequality]"
            arg_type = String
            default = "preventive_equality"
            range_tester = x -> x in ["frequency_ctrl", "preventive_penalty", "preventive_equality", "corrective_penalty", "corrective_equality", "corrective_inequality"]
            metavar = "CCON"
        "--load_scale"
            help = "Load multiplier"
            arg_type = Float64
            default = 1.0
            range_tester = x -> x >= 0
            metavar = "LSCALE"
        "--quad_penalty"
            help = "Qaudratic penalty parameter"
            arg_type = Float64
            default = 1e3
            range_tester = x -> x >= 0
            metavar = "QPEN"
        "--auglag_rho"
            help = "Aug Lag parameter"
            arg_type = Float64
            default = 1.0
            range_tester = x -> x >= 0
            metavar = "RHO"
        "--compute_mode"
            help = "Choose from: [nondecomposed, coldstart, lyapunov_bound]"
            arg_type = String
            default = "coldstart"
            range_tester = x -> x in ["nondecomposed", "coldstart", "lyapunov_bound"]
            metavar = "MODE"
    end

    return parse_args(ARGS, s)
end

function get_unique_name(dict::Dict)
    name = ""
    sorted_dict = sort(collect(dict), by=x->lowercase(x[1]))
    for entry in sorted_dict
        key_type = string(entry[1])
        key_val = string(entry[2])
        name *= key_type * "=" * key_val * "_"
    end
    return name
end


