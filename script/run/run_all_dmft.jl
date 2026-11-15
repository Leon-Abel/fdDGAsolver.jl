"""
run_all_dmft.jl

Batch driver: run `run_hubbard_model.jl` for the DMFT reference files
`data/dmft_U<U>_T<T>.h5`, optionally restricted to ranges of U and T.

`run_hubbard_model.jl` performs exactly one run per invocation, selected by
`--strategy` (scPA | fdPA — which reference vertex F0) and `--iteration-scheme`
(anderson | mfRG — how the fixed point is reached). This driver sweeps the cross
product

    selected (U, T) inputs  ×  strategies  ×  iteration_schemes

launching one fresh Julia process per combination. A failing / non-converging
combination is reported but does not abort the batch.

Usage (from the repo root):
  julia --project=script -t <nthreads> script/run/run_all_dmft.jl [options]

Run with `--help` for the full option list. The sweep lists take comma-separated
values; everything not passed stays at `run_hubbard_model.jl`'s own default, so
the defaults are written down in exactly one place.

  --strategies         comma-separated scPA | fdPA            (default fdPA)
  --schemes            comma-separated anderson | mfRG        (default both)
  --Us                 comma-separated U values / U ranges    (default all on disk)
  --Ts                 comma-separated T values / T ranges    (default all on disk)

`--Us` and `--Ts` select which of the files on disk to run; a file is run when
its U and its T are both selected. An entry is either a single value (`2.9`) or a
closed range `min:max` with either end optional (`2.0:2.12`, `2.8:`, `:1.0`);
entries are combined, so

  julia --project=script -t 8 script/run/run_all_dmft.jl --Us=0.5,2.0:2.12 --Ts=0.5

runs U = 0.5 together with every available U between 2.0 and 2.12, at T = 0.5.
U and T are read from the file name, not from the file, and an entry matching no
file is an error — reported up front rather than after the earlier (long) runs.

Examples:
  julia --project=script -t 8 script/run/run_all_dmft.jl --Ts=0.5
  julia --project=script -t 8 script/run/run_all_dmft.jl --nmax=12 --nq=6 --schemes=mfRG
  julia --project=script -t 8 script/run/run_all_dmft.jl --Us=5.0 --Ts=:2.0 --dry-run

Outputs land in `output/hubbard_model/`.
"""

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
nmax       = 10                  # half-box size for the vertex grids
nq         = 4                   # linear k-grid size for the vertex mesh
NL_method  = 1                   # vertex momentum structure: 1 (K2(k,q)≈K2(q) ...
threads    = Threads.nthreads()  # threads per spawned run (default: this driver's own)
strategies = "fdPA"              # comma-separated scPA | fdPA
schemes    = "mfRG"              # comma-separated anderson | mfRG
Us         = "2.2:2.7"           # comma-separated U values / `Umin:Umax` ranges (default ...
Ts         = ""                  # comma-separated T values / `Tmin:Tmax` ranges (default ...
dry_run    = false               # list the runs and exit without launching anything

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Run run_hubbard_model.jl across the DMFT inputs data/dmft_U<U>_T<T>.h5.")
    add_box_args!(settings; nmax, nq, NL_method)
    add_threads_arg!(settings; threads)
    add_arg_table!(settings,
        ["--strategies"], Dict(:help => "comma-separated scPA | fdPA",
                               :arg_type => String, :default => strategies),
        ["--schemes", "--iteration_schemes"],
                          Dict(:help => "comma-separated anderson | mfRG",
                               :arg_type => String, :default => schemes),
        ["--Us"],         Dict(:help => "comma-separated U values / `Umin:Umax` ranges " *
                                        "(default: every input on disk)",
                               :arg_type => String, :default => Us),
        ["--Ts"],         Dict(:help => "comma-separated T values / `Tmin:Tmax` ranges " *
                                        "(default: every input on disk)",
                               :arg_type => String, :default => Ts))
    add_flags!(settings, "dry-run" => "list the runs and exit without launching anything")

    parsed = parse_args(ARGS, settings)

    nmax       = parsed["nmax"]
    nq         = parsed["nq"]
    NL_method  = parsed["NL_method"]
    threads    = parsed["threads"]
    strategies = parsed["strategies"]
    schemes    = parsed["schemes"]
    Us         = parsed["Us"]
    Ts         = parsed["Ts"]
    dry_run    = parsed["dry_run"]
end

nthreads   = threads
# Validated here as well as in the child: a typo in a sweep list would otherwise
# only surface once that combination is reached, i.e. after the earlier (long)
# runs have already completed.
strategies = parse_strategies(strategies)
schemes    = parse_schemes(schemes)
U_ranges   = parse_U_ranges(Us)
T_ranges   = parse_U_ranges(Ts)   # same syntax for T

script = joinpath(@__DIR__, "run_hubbard_model.jl")

# ---------------------------------------------------------------------------
# Collect the DMFT inputs, keyed by the (U, T) in their file name
# ---------------------------------------------------------------------------
inputs = scan_dmft()   # sorted by U, then T; U and T as spelled in the name
isempty(inputs) && error("No dmft_U*_T*.h5 files found in $(data_dir())")

# An entry matching no file is a typo or a missing input file, not an empty sweep
# — say so before launching anything, and list what is available.
for (ranges, key, name) in ((U_ranges, :U, "U"), (T_ranges, :T, "T"))
    unmatched = [label for (lo, hi, label) in ranges
                 if !any(lo <= r[key] <= hi for r in inputs)]
    isempty(unmatched) || error(
        "No DMFT input for $name entry/entries $(join(unmatched, ", ")). " *
        "Available $name: $(join(unique(r[Symbol(name, "_str")] for r in inputs), ", ")).")
end

filter!(r -> in_U_ranges(r.U, U_ranges) && in_U_ranges(r.T, T_ranges), inputs)
isempty(inputs) && error("No DMFT input has both a selected U and a selected T. " *
                         "Available: $(join(("U$(r.U_str)_T$(r.T_str)" for r in scan_dmft()), ", ")).")

# One entry per process to launch. The input varies slowest so that all runs for
# a given input stay together in the log.
runs = [(r, strategy, scheme) for r in inputs
                              for strategy in strategies
                              for scheme in schemes]

println("Running run_hubbard_model.jl for $(length(inputs)) DMFT input(s), " *
        "$(length(runs)) run(s) total")
println("  nmax = $nmax, nq = $nq, NL_method = $NL_method")
println("  strategies = $(join(strategies, ", ")), iteration_schemes = $(join(schemes, ", "))")
println("  (U, T) = $(join(("($(r.U_str), $(r.T_str))" for r in inputs), ", "))")
println("  threads per run = $nthreads\n")

if dry_run
    for (i, (r, strategy, scheme)) in enumerate(runs)
        println("[$i/$(length(runs))] U = $(r.U_str), T = $(r.T_str) [$strategy/$scheme]")
    end
    println("\n--dry-run: nothing launched.")
    exit(0)
end

# ---------------------------------------------------------------------------
# Run each combination in a fresh Julia process
# ---------------------------------------------------------------------------
failures = String[]

for (i, (r, strategy, scheme)) in enumerate(runs)
    label = "U = $(r.U_str), T = $(r.T_str) [$strategy/$scheme]"
    println("[$i/$(length(runs))] === $label ===")

    # `U<U>_T<T>` is the `--dmft` shorthand for data/dmft_U<U>_T<T>.h5.
    args = child_args(; dmft = "U$(r.U_str)_T$(r.T_str)", nmax, nq, NL_method,
                        strategy, iteration_scheme = scheme)
    t0 = time()
    try
        run(julia_cmd(script, args; threads = nthreads))
        println("[$i/$(length(runs))] done in $(format_duration(time() - t0)): $label\n")
    catch err
        @warn "run_hubbard_model.jl failed for $label" exception = err
        push!(failures, label)
    end
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
println("\n=== Batch complete: $(length(runs) - length(failures))/$(length(runs)) succeeded ===")
if !isempty(failures)
    println("Failed runs:")
    foreach(f -> println("  - $f"), failures)
    exit(1)
end
