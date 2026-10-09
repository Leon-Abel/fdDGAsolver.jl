"""
run_all_high_temperature.jl

Batch driver: run `run_hubbard_model.jl` for the DMFT reference files in
`data/high_temperature_U<U>.h5`, optionally restricted to a range of U.

`run_hubbard_model.jl` performs exactly one run per invocation, selected by
`--strategy` (scPA | fdPA — which reference vertex F0) and `--iteration-scheme`
(anderson | mfRG — how the fixed point is reached). This driver sweeps the cross
product

    selected U  ×  strategies  ×  iteration_schemes

launching one fresh Julia process per combination. A failing / non-converging
combination is reported but does not abort the batch.

Usage (from the repo root):
  julia --project=script -t <nthreads> script/run/run_all_high_temperature.jl [options]

Run with `--help` for the full option list. The sweep lists take comma-separated
values; everything not passed stays at `run_hubbard_model.jl`'s own default, so
the defaults are written down in exactly one place.

  --strategies         comma-separated scPA | fdPA            (default fdPA)
  --schemes            comma-separated anderson | mfRG        (default both)
  --Us                 comma-separated U values / U ranges    (default all on disk)

`--Us` selects which of the files on disk to run. An entry is either a single
value (`2.9`) or a closed range `Umin:Umax` with either end optional (`2.0:2.12`,
`2.8:`, `:1.0`); entries are combined, so

  julia --project=script -t 8 script/run/run_all_high_temperature.jl --Us=0.5,2.0:2.12

runs U = 0.5 together with every available U between 2.0 and 2.12. U is read from
the file name, not from the file, and an entry matching no file is an error —
reported up front rather than after the earlier (long) runs.

Examples:
  julia --project=script -t 8 script/run/run_all_high_temperature.jl
  julia --project=script -t 8 script/run/run_all_high_temperature.jl --nmax=12 --nq=6 --schemes=mfRG
  julia --project=script -t 8 script/run/run_all_high_temperature.jl --Us=2.8: --dry-run

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
schemes    = "anderson,mfRG"     # comma-separated anderson | mfRG
Us         = ""                  # comma-separated U values / `Umin:Umax` ranges (default ...
dry_run    = false               # list the runs and exit without launching anything

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Run run_hubbard_model.jl across the high-temperature DMFT inputs.")
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
                               :arg_type => String, :default => Us))
    add_flags!(settings, "dry-run" => "list the runs and exit without launching anything")

    parsed = parse_args(ARGS, settings)

    nmax       = parsed["nmax"]
    nq         = parsed["nq"]
    NL_method  = parsed["NL_method"]
    threads    = parsed["threads"]
    strategies = parsed["strategies"]
    schemes    = parsed["schemes"]
    Us         = parsed["Us"]
    dry_run    = parsed["dry_run"]
end

nthreads   = threads
# Validated here as well as in the child: a typo in a sweep list would otherwise
# only surface once that combination is reached, i.e. after the earlier (long)
# runs have already completed.
strategies = parse_strategies(strategies)
schemes    = parse_schemes(schemes)
U_ranges   = parse_U_ranges(Us)

script = joinpath(@__DIR__, "run_hubbard_model.jl")

# ---------------------------------------------------------------------------
# Collect the high-temperature DMFT inputs, keyed by the U in their file name
# ---------------------------------------------------------------------------
Us_on_disk = scan_dmft_U()   # sorted numerically, U as spelled in the name
isempty(Us_on_disk) && error("No high_temperature_U*.h5 files found in $(data_dir())")

if !isempty(U_ranges)
    # An entry matching nothing is a typo or a missing input file, not an empty
    # sweep — say so before launching anything, and list what is available.
    unmatched = [label for (lo, hi, label) in U_ranges
                 if !any(lo <= parse(Float64, u) <= hi for u in Us_on_disk)]
    isempty(unmatched) || error(
        "No high_temperature input for U entry/entries $(join(unmatched, ", ")). " *
        "Available U: $(join(Us_on_disk, ", ")).")

    filter!(u -> in_U_ranges(parse(Float64, u), U_ranges), Us_on_disk)
end

# One entry per process to launch. U varies slowest so that all runs for a given
# input stay together in the log.
runs = [(U, strategy, scheme) for U in Us_on_disk
                              for strategy in strategies
                              for scheme in schemes]

println("Running run_hubbard_model.jl for $(length(Us_on_disk)) high_temperature input(s), " *
        "$(length(runs)) run(s) total")
println("  nmax = $nmax, nq = $nq, NL_method = $NL_method")
println("  strategies = $(join(strategies, ", ")), iteration_schemes = $(join(schemes, ", "))")
println("  U = $(join(Us_on_disk, ", "))")
println("  threads per run = $nthreads\n")

if dry_run
    for (i, (U, strategy, scheme)) in enumerate(runs)
        println("[$i/$(length(runs))] U = $U [$strategy/$scheme]")
    end
    println("\n--dry-run: nothing launched.")
    exit(0)
end

# ---------------------------------------------------------------------------
# Run each combination in a fresh Julia process
# ---------------------------------------------------------------------------
failures = String[]

for (i, (U, strategy, scheme)) in enumerate(runs)
    label = "U = $U [$strategy/$scheme]"
    println("[$i/$(length(runs))] === $label ===")

    args = child_args(; dmft = U, nmax, nq, NL_method,
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
