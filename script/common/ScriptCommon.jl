"""
    ScriptCommon

Helpers shared by the scripts in `script/`: file paths and names, command-line
arguments, printing, convergence data, and plotting. Load it with

    include(joinpath(@__DIR__, "common", "ScriptCommon.jl"))
    using .ScriptCommon

This module does not load `fdDGAsolver`, so it loads quickly. That suits batch
scripts, which only start other runs and do no calculation themselves.

Scripts that build and solve a model also load [`ScriptRuns`](@ref):

    include(joinpath(@__DIR__, "common", "ScriptRuns.jl"))
    using .ScriptRuns
"""
module ScriptCommon

include("paths.jl")
include("cli.jl")
include("logging.jl")
include("convergence.jl")
include("plotting.jl")

# Names a script can use directly after `using .ScriptCommon`, grouped by file.

# --- paths.jl ----------------------------------------------------------------
export REPO_ROOT, repo_path, data_dir, output_dir
export STRATEGIES, ITERATION_SCHEMES, ALGO_FAMILY, STRATEGY_OF_FAMILY
export strategy_label, arg_symbol, check_run, check_NL
export run_basename, solver_dir, solver_path, RUN_FILE_RE, parse_run_basename
export anderson_depth_dir, anderson_depth_basename, anderson_depth_path, anderson_depth_re
export postprocess_dir, postprocess_path
export dmft_file, DMFT_FILE_RE, scan_dmft_U
export figures_dir, CHI_SCAN_METHODS, chi_scan_basename, chi_scan_path

# --- cli.jl ------------------------------------------------------------------
export cli_settings, add_box_args!, add_run_args!, add_convergence_args!
export add_anderson_args!, add_dmft_arg!, add_threads_arg!, add_flags!
export parse_ints, parse_floats, parse_symbols, parse_dmft_inputs, resolve_dmft
export parse_U_ranges, in_U_ranges, U_ATOL, parse_nls, parse_strategies, parse_schemes
export child_args, julia_cmd
# Also pass on these ArgParse functions, so scripts don't need `using ArgParse`.
export parse_args, add_arg_table!, ArgParseSettings

# --- logging.jl --------------------------------------------------------------
export @mainprintf, format_duration, print_table, write_tsv

# --- convergence.jl ----------------------------------------------------------
export save_convergence!, write_convergence!, load_convergence, report_convergence

# --- plotting.jl -------------------------------------------------------------
export selected, k_index, q_label, q_tex, nu0_row, omega0_row
export bare_chi_q, bare_chi_local, dmft_temperature, dmft_hopping, dmft_model_params
export scan_available, windowed_U, cache_residual

end # module
