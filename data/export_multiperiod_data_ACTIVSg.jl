import ExaPF: PowerSystem
import ExaPF.PowerSystem: ParseMAT
const PS = PowerSystem
using DelimitedFiles
using CSV, DataFrames
using Dates

"""
    export_mp_demand_data(
        matfile::String,
        pd_file_in::String,
        qd_file_in::String,
        renewable_file_in::String,
        pd_file_out::String,
        qd_file_out::String,
        pgmax_file_out::String,
        n_interpolate::Int,
        n_periods::Int,
        month::Int=1,
    )

Exports multiperiod Pd, Qd and (renewable) Pgmax data in a format that can be read by `ProxAL`.

`matfile` is the casefile in MATPOWER format.

`pd_file_in`, `qd_file_in` and `renewable_file_in` must be downloaded from:
https://electricgrids.engr.tamu.edu/activsg-time-series-data/

`pd_file_out`, `qd_file_out` and `pgmax_file_out` are filenames where outputs will be written.
Note that `pgmax_file_out` contains Pgmax for all generators.
For non-renewable units, these values are the same as the values in the case files.

`n_interpolate` is the number of additional data points (per hour) generated using linear interpolation.

`n_periods` determines the sizes of the output matrices that will be written to file.

`month` must be between `1` and `12` and values will be read starting from the first hour of that month.
"""
function export_mp_demand_data(
    matfile::String,
    pd_file_in::String,
    qd_file_in::String,
    renewable_file_in::String,
    pd_file_out::String,
    qd_file_out::String,
    pgmax_file_out::String,
    n_interpolate::Int,
    n_periods::Int,
    month::Int=1,
)
    # read MATPOWER case file
    data_mat = ParseMAT.parse_mat(matfile)
    data = ParseMAT.mat_to_exapf(data_mat)
    bus_arr = data["bus"]
    gen_arr = data["gen"]
    busIdx = PS.get_bus_id_to_indexes(bus_arr)
    BusGeners = PS.get_bus_generators(bus_arr, gen_arr, busIdx)
    nbus = length(busIdx)
    ngen = size(gen_arr, 1)

    # read all data as DataFrames
    pd_df = CSV.read(pd_file_in, DataFrame; header=2, normalizenames=true)
    qd_df = CSV.read(qd_file_in, DataFrame; header=2, normalizenames=true)
    pg_df = CSV.read(renewable_file_in, DataFrame; header=2, normalizenames=true)

    # some buses have multiple loads attached
    # so we map each column index to the busIdx where that particular load should be added
    pd_colIdx_to_bus_idx = zeros(Int64, size(pd_df, 2))
    col_names = names(pd_df)
    for (i, c) in enumerate(col_names)
        if startswith(c, "Bus_")
            bus_id = parse(Int, c[5:3+findfirst('_', c[5:end])])
            pd_colIdx_to_bus_idx[i] = busIdx[bus_id]
        end
    end

    # repeat for the renewable generation units
    pg_colIdx_to_gen_idx = zeros(Int64, size(pg_df, 2))
    col_names = names(pg_df)
    for (i, c) in enumerate(col_names)
        if startswith(c, "Gen_")
            bus_id = parse(Int, c[5:3+findfirst('_', c[5:end])])
            generator_number = parse(Int, c[findfirst('_', c[5:end])+5:findfirst('M', c)-2])
            pg_colIdx_to_gen_idx[i] = BusGeners[busIdx[bus_id]][generator_number]
        end
    end

    # pd, qd, pgmax output to write to file
    Pd_arr = zeros(nbus, n_periods)
    Qd_arr = zeros(nbus, n_periods)
    Pg_arr = zeros(ngen, n_periods)
    Pg_arr[:, :] .= gen_arr[:, 9] # initializing with Pgmax from case file

    # column indices of Pd_arr, Qd_arr
    # which will be read from pd_df, qd_df
    hourly_indices = 1:(n_interpolate+1):n_periods

    # t_init:t_last are the hourly entries to read from pd_df, qd_df, pg_df
    t_init = (Date(Dates.Year(2016), Dates.Month(month), Dates.Day(1))
              -
              Date(Dates.Year(2016), Dates.Month(1), Dates.Day(1))).value
    t_init *= 24
    t_init += 1
    t_last = length(hourly_indices)
    for (colIdx, bus_idx) in enumerate(pd_colIdx_to_bus_idx)
        if bus_idx > 0
            Pd_arr[bus_idx, hourly_indices] .+= pd_df[t_init:t_last+t_init-1, colIdx]
            Qd_arr[bus_idx, hourly_indices] .+= qd_df[t_init:t_last+t_init-1, colIdx]
        end
    end
    for (colIdx, gen_idx) in enumerate(pg_colIdx_to_gen_idx)
        if gen_idx > 0
            Pg_arr[gen_idx, hourly_indices] .= pg_df[t_init:t_last+t_init-1, colIdx]
        end
    end

    # interpolate missing values
    if n_interpolate > 0
        for j in 2:length(hourly_indices)
            col1 = hourly_indices[j-1]
            col2 = hourly_indices[j]

            Pd_delta = (Pd_arr[:, col2] .- Pd_arr[:, col1]) / (n_interpolate + 1)
            Qd_delta = (Qd_arr[:, col2] .- Qd_arr[:, col1]) / (n_interpolate + 1)
            Pg_delta = (Pg_arr[:, col2] .- Pg_arr[:, col1]) / (n_interpolate + 1)
            for (idx, c) in enumerate((col1+1):(col2-1))
                Pd_arr[:, c] .= Pd_arr[:, col1] .+ (idx * Pd_delta)
                Qd_arr[:, c] .= Qd_arr[:, col1] .+ (idx * Qd_delta)
                Pg_arr[:, c] .= Pg_arr[:, col1] .+ (idx * Pg_delta)
            end
        end
    end

    # write to file
    writedlm(pd_file_out, Pd_arr)
    writedlm(qd_file_out, Qd_arr)
    writedlm(pgmax_file_out, Pg_arr)
    nothing
end

function export_all()
    for case in ["case_ACTIVSg2000", "case_ACTIVSg10k"], month in 1:12
        for n_interpolate in [0, 1, 5],
            n_periods = 168 * (n_interpolate + 1)
            resolution = Int(60 / (n_interpolate + 1))
            export_mp_demand_data(
                "$(case).m",
                "ACTIVSg_Time_Series/$(case)_load_time_series_MW.csv",
                "ACTIVSg_Time_Series/$(case)_load_time_series_MVAR.csv",
                "ACTIVSg_Time_Series/$(case)_renewable_time_series_MW.csv",
                "ACTIVSg_Time_Series/mp_demand/$(case)_$(Dates.format(Date(Month(month)), "u"))_oneweek_$(n_periods)_$(resolution)min.Pd",
                "ACTIVSg_Time_Series/mp_demand/$(case)_$(Dates.format(Date(Month(month)), "u"))_oneweek_$(n_periods)_$(resolution)min.Qd",
                "ACTIVSg_Time_Series/mp_demand/$(case)_$(Dates.format(Date(Month(month)), "u"))_oneweek_$(n_periods)_$(resolution)min.Pgmax",
                n_interpolate,
                n_periods,
                month,
            )
        end
    end
end


export_all()
