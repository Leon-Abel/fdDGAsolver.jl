# =============================================================================
# Command-line arguments shared by all scripts.
#
# This file defines each option (name, help text, type, allowed values).
# Default values are NOT set here: each script sets its own defaults at the top
# and passes them in, e.g. `add_box_args!(settings; nmax, nq, NL_method)`.
# That way the script and its command line always use the same defaults.
#
# All arguments are named, so their order does not matter.
# =============================================================================


using ArgParse  # used here: ArgParseSettings, add_arg_table!, ArgParse.parse_item

# Accept both `--strategy=fdPA` and `--strategy=:fdPA` (the colon is dropped).
ArgParse.parse_item(::Type{Symbol}, x::AbstractString) = arg_symbol(x)

"""
    cli_settings(description; kwargs...) -> ArgParseSettings

Base settings for a script, including `--help`.

On the command line, use dashes instead of underscores: `--NL_method` is typed
as `--NL-method`. The value is still stored under `"NL_method"`.
"""
function cli_settings(description; kwargs...)
    return ArgParseSettings(;
        description,                # shown at the top of `--help`
        autofix_names   = true,     # `--NL_method` is typed as `--NL-method`
        add_help        = true,     # add the `--help` option
        exit_after_help = true,     # stop the script after printing `--help`
        kwargs...)                  # any other ArgParseSettings options
end

# -----------------------------------------------------------------------------
# Reusable argument groups
# -----------------------------------------------------------------------------
# Each function adds a group of related arguments. Default values must be
# passed in by the calling script.

"""
    add_box_args!(s; nmax, nq, NL_method)

Vertex grid size (`--nmax`, `--nq`) and momentum structure (`--NL`).

Pass `NL_method = nothing` to skip `--NL` (used by plot scripts, which take a
list of NL methods via their own `--nls` option).
"""
function add_box_args!(s; nmax, nq, NL_method)
    add_arg_table!(s,
        ["--nmax"], Dict(:help     => "half-box size for the vertex grids",    # text shown by `--help`
                         :arg_type => Int,                                     # convert the input to this type
                         :default  => nmax,                                    # value used if the option is not given
                         :range_tester => >(0)),                               # reject values that fail this test
        ["--nq"],   Dict(:help     => "linear k-grid size for the vertex mesh",
                         :arg_type => Int, :default => nq,
                         :range_tester => >(0)))                               # must be positive
    NL_method === nothing && return s
    add_arg_table!(s,
        ["--NL_method", "--NL"],    # full name, short alias
                    Dict(:help     => "vertex momentum structure: 1 (K2(k,q)≈K2(q), K3(k,k',q)≈K3(q)), " *
                                      "2 (K3(k,k',q)≈K3(q)), 3 (full momentum dependence)",
                         :arg_type => Int,
                         :default => NL_method,
                         :range_tester => in(1:3)))      # must be 1, 2 or 3
    return s
end

"""
    add_run_args!(s; strategy, iteration_scheme)

`--strategy` (reference vertex) and `--iteration_scheme` (solver method).
"""
function add_run_args!(s; strategy, iteration_scheme)
    add_arg_table!(s,
        ["--strategy"], Dict(:help     => "reference vertex F0: scPA (bare, parquet " *
                                          "approximation) or fdPA (DMFT vertex)",
                             :arg_type => Symbol,
                             :default => strategy,
                             :range_tester => in(STRATEGIES)),  # must be a known strategy
        ["--iteration_scheme", "--scheme"],
                        Dict(:help     => "how the fixed point is reached: anderson or preconditioning (mfRG)",
                             :arg_type => Symbol,
                             :default => iteration_scheme,
                             :range_tester => in(ITERATION_SCHEMES)))  # must be a known scheme
    return s
end

"""
    add_convergence_args!(s; tol, maxiter)

Solver stopping criteria: tolerance and maximum number of iterations.
"""
function add_convergence_args!(s; tol, maxiter)
    add_arg_table!(s,
        ["--tol"],     Dict(:help => "residual-norm convergence tolerance",
                            :arg_type => Float64, 
                            :default => tol,
                            :range_tester => >(0)),         # must be positive
        ["--maxiter"], Dict(:help => "maximum parquet iterations",
                            :arg_type => Int, 
                            :default => maxiter,
                            :range_tester => >(0)))         # must be positive
    return s
end

"""
    add_anderson_args!(s; mem, delta)

Anderson acceleration settings: history depth `--mem` and mixing `--delta`.
"""
function add_anderson_args!(s; mem, delta)
    add_arg_table!(s,
        ["--mem"], Dict(:help => "Anderson history depth (0 = damped Picard)",
                        :arg_type => Int,
                        :default => mem,
                        :range_tester => >=(0)),            # 0 or more
        ["--delta"], Dict(:help => "Anderson mixing β handed to NLsolve",
                          :arg_type => Float64, 
                          :default => delta,
                          :range_tester => d -> 0 < d <= 1))  # between 0 (excluded) and 1
    return s
end

"""
    add_dmft_arg!(s; default)

The DMFT input file: a path relative to the repo root, or just a U value as
shorthand for `data/high_temperature_U<U>.h5`.
"""
function add_dmft_arg!(s; default)
    add_arg_table!(s,
        ["--dmft"], Dict(:help    => "DMFT reference: a path relative to the repo root, " *
                                     "or just the U (shorthand for data/high_temperature_U<U>.h5)",
                         :arg_type => String, 
                         :default => default))  # no check here; see `resolve_dmft`
    return s
end

"""
    add_threads_arg!(s; threads)

Number of threads for each run started by a batch script.
"""
function add_threads_arg!(s; threads)
    add_arg_table!(s,
        ["--threads"], Dict(:help => "threads for each started run (default: same as this batch script)",
                            :arg_type => Int, 
                            :default => threads,
                            :range_tester => >(0)))         # must be positive
    return s
end

"""
    add_flags!(s, "name" => "help text", ...)

Add yes/no switches that take no value. Each switch is off (`false`) unless it
is typed on the command line, which turns it on (`true`).

Example: `add_flags!(settings, "dry-run" => "list the runs and exit")`, then
`julia script.jl --dry-run` gives `parsed["dry_run"] == true`.
"""
function add_flags!(s, flags::Pair{String,String}...)
    for (name, help) in flags
        add_arg_table!(s, ["--$name"], Dict(:help => help, :action => :store_true))  # false unless the flag is given
    end
    return s
end

# -----------------------------------------------------------------------------
# List-valued arguments
# -----------------------------------------------------------------------------
# Lists are passed as one comma-separated string, e.g. `--mems=0,4,8`,
# and parsed by the functions below.

"""
    parse_ints(s) -> Vector{Int}

Comma-separated integers; ranges `lo:hi` and `lo:step:hi` are allowed.
Duplicates are removed and the result is sorted: `"0,1:4,16"` -> `[0, 1, 2, 3, 4, 16]`.
"""
function parse_ints(s)
    out = Int[]
    for part in split(s, ',')
        item = strip(part)
        isempty(item) && continue
        if occursin(':', item)
            bounds = parse.(Int, strip.(split(item, ':')))
            r = length(bounds) == 2 ? (bounds[1]:bounds[2])            :
                length(bounds) == 3 ? (bounds[1]:bounds[2]:bounds[3])  :
                error("Range must be `lo:hi` or `lo:step:hi`, got `$item`.")
            isempty(r) && error("Range `$item` is empty.")
            append!(out, r)
        else
            push!(out, parse(Int, item))
        end
    end
    isempty(out) && error("Expected at least one integer, got `$s`.")
    return sort!(unique!(out))
end

"""
    parse_floats(s) -> Vector{Float64}

Comma-separated numbers. Duplicates are removed and the result is sorted.
"""
function parse_floats(s)
    out = [parse(Float64, strip(p)) for p in split(s, ',') if !isempty(strip(p))]
    isempty(out) && error("Expected at least one number, got `$s`.")
    return sort!(unique!(out))
end

"""
    parse_symbols(s) -> Vector{Symbol}

Comma-separated symbols; the leading colon is optional: `"scPA,:fdPA"` -> `[:scPA, :fdPA]`.
"""
function parse_symbols(s)
    out = [arg_symbol(strip(p)) for p in split(s, ',') if !isempty(strip(p))]
    isempty(out) && error("Expected at least one value, got `$s`.")
    return unique!(out)
end

"""
    parse_dmft_inputs(s) -> Vector{String}

Comma-separated DMFT inputs, as paths relative to the repo root. A plain
number U is shorthand for `data/high_temperature_U<U>.h5`.
"""
function parse_dmft_inputs(s)
    out = String[]
    for part in split(s, ',')
        item = String(strip(part))
        isempty(item) && continue
        push!(out, occursin('/', item) || occursin('\\', item) || endswith(item, ".h5") ?
                   item : "data/high_temperature_U$(item).h5")
    end
    isempty(out) && error("Expected at least one DMFT input, got `$s`.")
    return out
end

"""
    resolve_dmft(arg) -> String

Full path of the DMFT input given by `--dmft`. Errors right away if the file
does not exist, so a typo is caught before the run starts.
"""
function resolve_dmft(arg)
    name = only(parse_dmft_inputs(arg))
    path = isabspath(name) ? name : repo_path(name)
    isfile(path) || error("DMFT reference data not found:\n  $path")
    return path
end

"""
    parse_U_ranges(s) -> Vector{Tuple{Float64,Float64,String}}

Comma-separated U values or ranges, returned as `(lo, hi, label)`.
Examples: `2.9`, `2.0:2.12`, `2.8:` (2.8 and up), `:1.0` (up to 1.0).
A single value matches within a small tolerance `U_ATOL` to allow for rounding.
`label` is the entry as typed, used in error messages.
"""
const U_ATOL = 1e-8

function parse_U_ranges(s)
    ranges = Tuple{Float64, Float64, String}[]
    for part in split(s, ',')
        item = String(strip(part))
        isempty(item) && continue
        if occursin(':', item)
            bounds = split(item, ':')
            length(bounds) == 2 || error("U range must be `Umin:Umax`, got `$item`.")
            lo_s, hi_s = strip(bounds[1]), strip(bounds[2])
            # A missing bound means no limit on that side.
            lo = isempty(lo_s) ? -Inf : parse(Float64, lo_s)
            hi = isempty(hi_s) ?  Inf : parse(Float64, hi_s)
            lo <= hi || error("U range must have Umin <= Umax, got `$item`.")
            push!(ranges, (lo, hi, item))
        else
            u = parse(Float64, item)
            push!(ranges, (u - U_ATOL, u + U_ATOL, item))
        end
    end
    return ranges
end

"True if `u` is in any of the ranges, or if no ranges are given."
in_U_ranges(u, ranges) = isempty(ranges) || any(lo <= u <= hi for (lo, hi, _) in ranges)

"""
    parse_nls(s) -> Vector{Int}

Comma-separated NL methods; each must be 1, 2 or 3.
"""
function parse_nls(s)
    nls = parse_ints(s)
    all(in(1:3), nls) || error("NL method must be 1, 2 or 3, got $(filter(!in(1:3), nls)).")
    return nls
end

"""
    parse_strategies(s) 

Comma-separated strategies, checked against the allowed values.
"""
function parse_strategies(s)
    v = parse_symbols(s)
    all(in(STRATEGIES), v) || error("strategy must be one of $(STRATEGIES), got $(filter(!in(STRATEGIES), v)).")
    return v
end

"""
    parse_schemes(s)

Comma-separated iteration schemes, checked against the allowed values.
"""
function parse_schemes(s)
    v = parse_symbols(s)
    all(in(ITERATION_SCHEMES), v) ||
        error("iteration_scheme must be one of $(ITERATION_SCHEMES), got $(filter(!in(ITERATION_SCHEMES), v)).")
    return v
end

# -----------------------------------------------------------------------------
# Starting runs from a batch script
# -----------------------------------------------------------------------------
# A batch script (e.g. run_all_high_temperature.jl) does no calculation itself:
# it starts a single-run script (e.g. run_hubbard_model.jl) once for each set
# of settings, each in its own Julia process.

"""
    child_args(; kwargs...) -> Vector{String}

Turn keyword arguments into command-line arguments for a run started by a batch
script: `child_args(nmax = 12, dry_run = true)` gives `["--nmax=12", "--dry-run"]`.

`true` becomes a plain flag; `false` and `nothing` are left out, so the started
run uses its own default. Underscores become dashes, since the command line only
accepts dashes.
"""
function child_args(; kwargs...)
    out = String[]
    for (k, v) in kwargs
        v === nothing && continue
        v === false  && continue
        name = "--" * replace(String(k), '_' => '-')
        v === true ? push!(out, name) : push!(out, "$name=$v")
    end
    return out
end

"""
    julia_cmd(script, args; threads, project) -> Cmd

Command to run `script` in a new Julia process. Each run gets its own process
so its memory is freed when it finishes (a loaded solver can be very large).
"""
function julia_cmd(script, args::AbstractVector{<:AbstractString};
                   threads = Threads.nthreads(), project = joinpath(REPO_ROOT, "script"))
    return `$(Base.julia_cmd()) --project=$project -t $threads $script $args`
end
