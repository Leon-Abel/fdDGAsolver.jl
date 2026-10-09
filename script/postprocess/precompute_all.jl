"""
precompute_all.jl

Batch driver: run `precompute_sigma_chi.jl` for every converged solver saved under
`output/hubbard_model/`, refreshing the Σ / χ cache of each of them in
`output/postprocess/`.

The runs are discovered from the solver file names, so this driver needs no list of
U values, box sizes or (strategy, iteration_scheme) pairs — it processes whatever
`run_hubbard_model.jl` has left on disk. Caches that already exist are overwritten,
which makes this the script to run after anything that changes what `precompute`
computes or stores; `--keep-existing` turns that off and fills in only what is
missing, which is how an interrupted batch is resumed.

Each run is launched in a fresh Julia process, as `run_all_high_temperature.jl` does:
one saved solver is several hundred MB on disk and a multiple of that once loaded, so
sweeping dozens of them inside a single process would pile up heap that no GC pause is
obliged to hand back. A failing run is reported but does not abort the batch.

Usage (from the repo root):
  julia --project=script -t <nthreads> script/postprocess/precompute_all.jl [--dry-run] [--keep-existing]

  --dry-run        list the runs that would be precomputed, then exit without
                   launching or overwriting anything
  --keep-existing  skip runs whose cache is already present, instead of
                   recomputing it

The number of threads used for each run matches the threads of this driver (set
via `julia -t <nthreads>`), or `--threads`. For a single run, call
`precompute_sigma_chi.jl` directly — it takes the same knobs as named options.
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
threads       = Threads.nthreads()  # threads per spawned run (default: this driver's own)
dry_run       = false               # list the runs that would be precomputed, then exit
keep_existing = false               # skip runs whose cache is already present, instead of ...

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings(
        "Precompute the Σ / χ caches for every converged solver saved under output/hubbard_model/.")
    add_threads_arg!(settings; threads)
    add_flags!(settings,
        "dry-run"       => "list the runs that would be precomputed, then exit",
        "keep-existing" => "skip runs whose cache is already present, instead of recomputing it")

    parsed        = parse_args(ARGS, settings)

    threads       = parsed["threads"]
    dry_run       = parsed["dry_run"]
    keep_existing = parsed["keep_existing"]
end
nthreads      = threads

solvers = solver_dir()
script  = joinpath(@__DIR__, "precompute_sigma_chi.jl")

isdir(solvers) || error("No solver directory at\n  $solvers\nNothing to precompute.")

# ---------------------------------------------------------------------------
# Discover the runs from the saved solver file names
# ---------------------------------------------------------------------------
# `parse_run_basename` (common/paths.jl) is the inverse of the `run_basename`
# that wrote these names, so the two cannot drift. Nothing here loads
# fdDGAsolver: this driver only spawns other processes, and ScriptCommon's light
# half is all it needs to match names and forward arguments.

# The cache shares the solver basename, and is only ever moved into place once
# fully written, so `--keep-existing` can tell a complete run from an incomplete
# one without opening anything.

runs       = NamedTuple[]
unreadable = String[]     # named like a solver, but not by knobs this repo knows

for f in readdir(solvers)
    r = parse_run_basename(f)
    r === nothing && continue

    if r.strategy === nothing || r.iteration_scheme === nothing
        push!(unreadable, f)
        continue
    end

    # `U_str` keeps U exactly as the file name spells it; `U` is the number, used
    # only for ordering. Rebuilding the name from the number would turn a run
    # saved as `U2.10` into a lookup for `U2.1`.
    push!(runs, (; r.file, r.U, r.U_str, r.T, r.nmax, r.nq, r.NL,
                   strategy = r.strategy, scheme = r.iteration_scheme))
end

isempty(runs) && error("No solver files matching hubbard_*.h5 found in\n  $solvers")

# `precompute_sigma_chi.jl` takes no T — it reads T back from the DMFT file for U —
# so two solver files differing only in their T address the very same child run. Keep
# one per address and name the collisions, rather than spending a second long run on
# a file the child was never going to open.
address(r) = (r.U, r.nmax, r.nq, r.NL, r.strategy, r.scheme)

kept       = Dict{Tuple, NamedTuple}()
duplicates = Tuple{String, String}[]   # (file not run, file run in its place)

for r in runs
    a = address(r)
    if haskey(kept, a)
        push!(duplicates, (r.file, kept[a].file))
    else
        kept[a] = r
    end
end

# Runs of one box size stay together, ascending in U, so the log reads in the order
# the plots are made from.
runs = sort!(collect(values(kept));
             by = r -> (r.nmax, r.nq, r.NL, String(r.strategy), String(r.scheme), r.U))

# `precompute` takes its reference from data/high_temperature_U<U>.h5, with U as the
# solver file name spells it — the same name `dmft_file(r.U_str)` builds here. A
# solver grown from any other DMFT input (Wu_point, weak_coupling) has no reference
# to subtract and cannot be precomputed at all — report those instead of launching
# a child that is certain to fail on the missing file.
has_dmft(r) = isfile(dmft_file(r.U_str))
no_dmft     = filter(!has_dmft, runs)
runs        = filter(has_dmft, runs)

complete(r) = isfile(joinpath(postprocess_dir(), r.file))
cached      = keep_existing ? filter(complete, runs) : NamedTuple[]
keep_existing && filter!(!complete, runs)

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
println("Precomputing self-energy and susceptibility caches for $(length(runs)) saved solver(s)")
println("  solvers         : $solvers")
println("  existing caches : ", keep_existing ? "kept (only missing ones are computed)" : "OVERWRITTEN")
println("  threads per run : $nthreads")

if !isempty(cached)
    println("\nAlready complete, skipped ($(length(cached))):")
    foreach(r -> println("  - $(r.file)"), cached)
end

if !isempty(no_dmft)
    println("\nNo DMFT reference data/high_temperature_U<U>.h5, skipped ($(length(no_dmft))):")
    foreach(r -> println("  - $(r.file)"), no_dmft)
end

if !isempty(duplicates)
    println("\nSame run under a different T, skipped ($(length(duplicates))):")
    for (dup, orig) in duplicates
        println("  - $dup (covered by $orig)")
    end
end

if !isempty(unreadable)
    println("\nUnrecognized strategy/iteration_scheme in the name, skipped ($(length(unreadable))):")
    foreach(f -> println("  - $f"), unreadable)
end

if isempty(runs)
    println("\nNothing to do.")
    exit(0)
end

println("\nRuns:")
foreach(r -> println("  - $(r.file)"), runs)

if dry_run
    println("\n--dry-run: nothing was computed.")
    exit(0)
end

# ---------------------------------------------------------------------------
# Run each solver through precompute_sigma_chi.jl in a fresh Julia process
# ---------------------------------------------------------------------------
failures = String[]

println()
for (i, r) in enumerate(runs)
    println("[$i/$(length(runs))] === $(r.file) ===")

    # U goes over as the string the file name spells it with: that is what the
    # child rebuilds both the DMFT file name and the solver file name from.
    # `--force` unless the caller asked to keep what is already there.
    args = child_args(; U = r.U_str, nmax = r.nmax, nq = r.nq, NL_method = r.NL,
                        strategies = r.strategy, schemes = r.scheme,
                        force = !keep_existing)
    try
        run(julia_cmd(script, args; threads = nthreads))
        println("[$i/$(length(runs))] done: $(r.file)\n")
    catch err
        @warn "precompute_sigma_chi.jl failed for $(r.file)" exception = err
        push!(failures, r.file)
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
