# =============================================================================
# Folder locations, run settings, and the file names runs are saved under.
#
# This file does not load `fdDGAsolver`, so batch scripts can use it without
# the slow package load.
# =============================================================================

"Full path of the repository folder."
const REPO_ROOT = rstrip(normpath(joinpath(@__DIR__, "..", "..")), ('/', '\\'))

# Path inside the repository, e.g. `repo_path("data", "file.h5")`.
repo_path(parts...) =normpath(joinpath(REPO_ROOT, parts...))

data_dir()    = repo_path("data")
output_dir()  = repo_path("output")
figures_dir() = repo_path("figures")

# -----------------------------------------------------------------------------
# Run settings
# -----------------------------------------------------------------------------
# A run is described by two settings:
#
#   strategy         :scPA | :fdPA      — which reference vertex F0 the solver uses
#   iteration_scheme :anderson | :mfRG  — how the solver iterates
#
# File names use the short forms "sc" / "fd" for the strategy.

const STRATEGIES        = (:scPA, :fdPA)
const ITERATION_SCHEMES = (:anderson, :mfRG)

# Strategy -> short form in file names, and back.
const ALGO_FAMILY         = Dict(:scPA => "sc", :fdPA => "fd")
const STRATEGY_OF_FAMILY  = Dict(v => k for (k, v) in ALGO_FAMILY)

"Name of a strategy for plot legends and printed output."
strategy_label(strategy) = strategy === :fdPA ? "pDΓA" : "PA"

"""
    arg_symbol(s) -> Symbol

Turn a string into a symbol, dropping a leading colon: both `"fdPA"` and
`":fdPA"` give `:fdPA`.
"""
arg_symbol(s) = Symbol(lstrip(String(s), ':'))

"Error if `strategy` or `iteration_scheme` is not a known value."
function check_run(strategy, iteration_scheme)
    strategy         in STRATEGIES        || error("strategy must be one of $(STRATEGIES), got :$strategy.")
    iteration_scheme in ITERATION_SCHEMES || error("iteration_scheme must be one of $(ITERATION_SCHEMES), got :$iteration_scheme.")
    return nothing
end

# Error if `NL_method` is not 1, 2 or 3.
check_NL(NL_method) =
    NL_method in 1:3 ? nothing : error("NL_method must be 1, 2 or 3, got $NL_method.")

# -----------------------------------------------------------------------------
# File names
# -----------------------------------------------------------------------------
# U and T are put into file names exactly as given (e.g. `2.0875` stays `2.0875`).

"""
    run_basename(U, T, nmax, nq, NL_method, strategy, iteration_scheme) -> String

File name of a run. The saved solver (`output/hubbard_model/`) and its
precompute cache (`output/postprocess/`) use the same name.
"""
run_basename(U, T, nmax, nq, NL_method, strategy, iteration_scheme) =
    "hubbard_$(ALGO_FAMILY[strategy])_$(iteration_scheme)_U$(U)_T$(T)" *
    "_nmax$(nmax)_nq$(nq)_NL$(NL_method).h5"

"Folder where solvers are saved."
solver_dir() =joinpath(output_dir(), "hubbard_model")

"Full path of the saved solver for one run."
solver_path(a::Vararg{Any,7}) = joinpath(solver_dir(), run_basename(a...))

"""
Pattern that matches file names made by `run_basename`, and picks out
`(family, scheme, U, T, nmax, nq, NL)` as strings. Used to find runs by name.
If you change `run_basename`, change this too.
"""
const RUN_FILE_RE = r"^hubbard_([a-z]+)_([A-Za-z]+)_U([0-9]*\.?[0-9]+)_T([0-9]*\.?[0-9]+)_nmax([0-9]+)_nq([0-9]+)_NL([0-9]+)\.h5$"

"""
    parse_run_basename(name) -> NamedTuple or nothing

Read the run settings back from a file name (the reverse of [`run_basename`](@ref)).

Returns `nothing` if `name` is not a run file. If the strategy or scheme in the
name is unknown, that field is `nothing`.
"""
function parse_run_basename(name)
    m = match(RUN_FILE_RE, name)
    m === nothing && return nothing
    fam, scheme = m.captures[1], Symbol(m.captures[2])
    return (; file = name,
              U_str = m.captures[3], T_str = m.captures[4],
              U    = parse(Float64, m.captures[3]),
              T    = parse(Float64, m.captures[4]),
              nmax = parse(Int,     m.captures[5]),
              nq   = parse(Int,     m.captures[6]),
              NL   = parse(Int,     m.captures[7]),
              strategy         = get(STRATEGY_OF_FAMILY, fam, nothing),
              iteration_scheme = scheme in ITERATION_SCHEMES ? scheme : nothing)
end

# -----------------------------------------------------------------------------
# Anderson-depth scan
# -----------------------------------------------------------------------------
# These runs are saved in their own folder, and their file names also include
# the Anderson settings `mem` and `delta`.

anderson_depth_dir() = joinpath(output_dir(), "anderson_depth")

anderson_depth_basename(mem, δ, U, T, nmax, nq, NL_method) =
    "hubbard_fd_anderson_mem$(mem)_delta$(δ)_U$(U)_T$(T)" *
    "_nmax$(nmax)_nq$(nq)_NL$(NL_method).h5"

anderson_depth_path(a::Vararg{Any,7}) =
    joinpath(anderson_depth_dir(), anderson_depth_basename(a...))

"""
Pattern that matches Anderson-depth file names for the given `nmax`, `nq` and
`NL_method`, and picks out `mem`, `delta` and `U`. Any U and T match.
"""
anderson_depth_re(nmax, nq, NL_method; mem = raw"\d+", delta = "[0-9.]+") =
    Regex("^hubbard_fd_anderson_mem($mem)_delta($delta)_U([0-9.]+)_T[0-9.]+" *
          "_nmax$(nmax)_nq$(nq)_NL$(NL_method)\\.h5\$")

# -----------------------------------------------------------------------------
# Precompute cache
# -----------------------------------------------------------------------------
# One file per run with the computed Σ and χ (see precompute_sigma_chi.jl).
# If the file exists, the run has been precomputed.

postprocess_dir() = joinpath(output_dir(), "postprocess")

"""
    postprocess_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme) -> String

Path of the precompute cache file for one run.
"""
postprocess_path(a::Vararg{Any,7}) = joinpath(postprocess_dir(), run_basename(a...))

# -----------------------------------------------------------------------------
# DMFT reference input
# -----------------------------------------------------------------------------

"Path of the DMFT input file for interaction `U`."
dmft_file(U) = joinpath(data_dir(), "high_temperature_U$(U).h5")

# Pattern that matches DMFT file names and picks out U.
const DMFT_FILE_RE =r"^high_temperature_U([0-9]*\.?[0-9]+)\.h5$"

"""
    scan_dmft_U() -> Vector{String}

All U values that have a `data/high_temperature_U<U>.h5` file, sorted from
smallest to largest. Returned as strings, written as in the file name.
"""
function scan_dmft_U()
    d = data_dir()
    isdir(d) || return String[]
    us = String[m.captures[1] for m in (match(DMFT_FILE_RE, f) for f in readdir(d)) if m !== nothing]
    return sort!(unique!(us); by = u -> parse(Float64, u))
end

# -----------------------------------------------------------------------------
# AIM susceptibility scan (calculate_susceptibilities.jl -> plot_susceptibilities.jl)
# -----------------------------------------------------------------------------
# A temperature scan of the Anderson impurity model (not a Hubbard run), so it
# uses its own file names.

"Methods a χ(T) scan can use, as written in the file name."
const CHI_SCAN_METHODS = (:sc, :fd, :HA)

"""
    chi_scan_basename(method, nmax, T_lo, T_hi, nT) -> String
    chi_scan_path(method, nmax, T_lo, T_hi, nT)     -> String

File name / full path of the χ(T) scan for one method (`:sc`, `:fd` or `:HA`).
"""
function chi_scan_basename(method, nmax, T_lo, T_hi, nT)
    method in CHI_SCAN_METHODS ||
        error("method must be one of $(CHI_SCAN_METHODS), got :$method.")
    return "Chi_$(method)_$(nmax)_$(T_lo)_$(T_hi)_$(nT).h5"
end

chi_scan_path(a::Vararg{Any,5}) = joinpath(output_dir(), chi_scan_basename(a...))
